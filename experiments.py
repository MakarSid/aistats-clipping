from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.optimize import brentq
from scipy.stats import t as student_t
from sklearn.datasets import load_diabetes

matplotlib.use("Agg")

SIGMA0_LEVELS = (0.0, 0.01, 0.1, 0.5, 1.0)
CLIPPING_LEVELS = (0.1, 0.5, 1.0, 4.0)
SSTM_SIGMA0 = (0.0, 1e-5, 1e-4, 1e-3)
SSTM_SIGMA1 = (0.0, 1e-4, 3e-4, 1e-3)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
STYLES = ("-", "--", "-.", ":", (0, (5, 1)))


@dataclass(frozen=True)
class Noise:
    alpha: float = 1.5
    sigma0: float = 0.05
    sigma1: float = 0.001


def pareto_noise(rng, size, alpha, symmetric=False):
    p = np.exp(rng.standard_exponential(size) / alpha)
    if symmetric:
        return p * (2 * rng.integers(0, 2, size=size, dtype=np.int8) - 1)
    mean = alpha / (alpha - 1)
    return (p - mean) / max(1.0, mean - 1)


def noise_scale(magnitude, noise):
    return (noise.sigma0**noise.alpha
            + (noise.sigma1 * np.abs(magnitude))**noise.alpha)**(1 / noise.alpha)


def sgd_parameters(n, noise, beta=0.05, L=1.0, R=1.0):
    """Theorem 5.1"""
    alpha, s0, s1 = noise.alpha, noise.sigma0, noise.sigma1
    ell = math.log(4 * n / beta)
    denominator = n * s0**alpha + 2**(alpha / 2) * s1**alpha * (L * R)**alpha
    ratio = n * (3 * math.sqrt(2) * L * R)**alpha / denominator if denominator else 0.0
    tail_ratio = max(80 * n, ratio)
    b = 2**(alpha - 1) / (alpha - 1)
    power = 2 / (2 - alpha) if alpha < 2 else math.inf
    v = 1.5**(2 - alpha) * min(power, 1 + 2 / alpha * math.log1p(tail_ratio))
    c = 180 * max(v, b / 3)
    base = c * 40**(alpha - 1) * ell**(alpha - 1)
    gamma_L = 1 / (80 * math.sqrt(2) * L * ell)
    gamma_0 = R / ((base * n)**(1 / alpha) * s0) if s0 else math.inf
    gamma_1 = (1 / ((base * 2**alpha)**(2 / alpha) * s1**2 * L
                    * n**((2 - alpha) / alpha))) if s1 else math.inf
    gamma_2 = (1 / ((base * 2**(alpha / 2))**(1 / alpha) * s1 * L)) if s1 else math.inf
    gamma = min(gamma_L, gamma_0, gamma_1, gamma_2)
    return {"gamma": gamma, "lambda": R / (40 * gamma * ell),
            "bound": 2 * R**2 / (gamma * n), "ell": ell,
            "gamma_L": gamma_L, "gamma_0": gamma_0,
            "gamma_1": gamma_1, "gamma_2": gamma_2}


def sstm_parameters(n, noise, beta=0.05, L=1.0, R=1.0):
    """Theorem 9"""
    alpha, s0, s1 = noise.alpha, noise.sigma0, noise.sigma1
    H = math.log(4 * n / beta)
    r = (alpha - 1) / alpha
    b = 2**(alpha - 1) / (alpha - 1)
    denominator = (s0 * (n + 1) / (17640 * L * R * H))**alpha + (s1 / 3)**alpha
    tail_ratio = max(200 * b * n, 1 / denominator if denominator else 0.0)
    power = 2 / (2 - alpha) if alpha < 2 else math.inf
    v = 1.5**(2 - alpha) * min(power, 1 + 2 / alpha * math.log1p(tail_ratio))
    q = min(1 / (1350 * v * 30**(alpha - 2)), 1 / (15 * b * 30**(alpha - 1)))
    c0, c1 = (2 / q)**(1 / alpha), 49 * (4 / q)**(2 / alpha)
    a_L = 176400 * H**2
    a_0 = c0 * s0 / (L * R) * (n + 1) * n**(1 / alpha) * H**r
    a_1 = c1 * s1**2 * n**(2 / alpha) * H**(2 * r)
    a = max(a_L, a_0, a_1)
    return {"a": a, "H": H, "a_L": a_L, "a_0": a_0, "a_1": a_1,
            "bound": 6 * a * L * R**2 / (n * (n + 3))}


def simulate_quadratic(n, runs, checkpoints, sigma0, sigma1, alpha,
                       gamma, lambdas, seed, symmetric=False):
    sigma0 = np.asarray(sigma0, dtype=float).reshape(-1, 1)
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1, 1)
    count = max(len(sigma0), len(lambdas))
    if len(sigma0) not in (1, count) or len(lambdas) not in (1, count):
        raise ValueError("Incompatible noise levels and clipping thresholds.")
    rng = np.random.default_rng(seed)
    x = np.ones((count, runs))
    total = np.zeros_like(x)
    gaps = np.empty((count, len(checkpoints), runs))
    sigma0_power = sigma0**alpha
    j = 0
    for k in range(n):
        total += x
        z = pareto_noise(rng, runs, alpha, symmetric)
        scale = (sigma0_power + (sigma1 * np.abs(x))**alpha)**(1 / alpha)
        x -= gamma * np.clip(x + scale * z, -lambdas, lambdas)
        if k + 1 == checkpoints[j]:
            gaps[:, j] = 0.5 * (total / (k + 1))**2
            j += 1
            if j == len(checkpoints):
                break
    return gaps


def mean_clipped_gradient(x, lam, noise):
    alpha = noise.alpha
    mean = alpha / (alpha - 1)
    scale = float(noise_scale(x, noise)) / max(1.0, mean - 1)
    if scale == 0:
        return float(np.clip(x, -lam, lam))
    offset = x - scale * mean
    upper = (lam - offset) / scale
    lower = (-lam - offset) / scale
    right_excess = scale * upper**(1 - alpha) / (alpha - 1) if upper > 1 else x - lam
    left_excess = (scale * (lower - 1 - (1 - lower**(1 - alpha)) / (alpha - 1))
                   if lower > 1 else 0.0)
    return x - right_excess + left_excess


def mean_field_root(lam, noise):
    field = lambda x: mean_clipped_gradient(x, lam, noise)
    if field(0) == 0:
        return 0.0
    if field(0) * field(1) > 0:
        return None
    return float(brentq(field, 0, 1, xtol=1e-14))


def regression_problem(rho=0.01):
    X, y = load_diabetes(return_X_y=True, scaled=False)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = float(y.mean()), float(y.std())
    A, target = (X - X_mean) / X_std, (y - y_mean) / y_std
    H = A.T @ A / len(y) + rho * np.eye(A.shape[1])
    w_star = np.linalg.solve(H, A.T @ target / len(y))
    eigenvalues, U = np.linalg.eigh(H)
    pivots = np.argmax(np.abs(U), axis=0)
    U *= np.where(U[pivots, np.arange(len(eigenvalues))] < 0, -1.0, 1.0)
    return {"A": A, "y": target, "H": H, "w_star": w_star,
            "eigenvalues": eigenvalues, "eigenvectors": U, "q_star": U.T @ w_star,
            "L": float(eigenvalues[-1]), "R": float(np.linalg.norm(w_star)),
            "X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def simulate_regression(n, runs, problem, noise, gamma, lambdas, seed, scale_model):
    h, star = problem["eigenvalues"], problem["q_star"]
    lambdas = np.asarray(lambdas).reshape(-1, 1)
    q = np.zeros((len(lambdas), runs, len(h)))
    total = np.zeros_like(q)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        total += q
        gradient = h * (q - star)
        magnitude = (np.abs(gradient[..., 0]) if scale_model == "projected"
                     else np.linalg.norm(gradient, axis=-1))
        gradient[..., 0] += noise_scale(magnitude, noise) * pareto_noise(rng, runs, noise.alpha)
        norms = np.linalg.norm(gradient, axis=-1)
        factors = np.minimum(1.0, np.divide(lambdas, norms,
                                          out=np.ones_like(norms), where=norms > 0))
        q -= gamma * factors[..., None] * gradient
    return 0.5 * np.sum(h * (total / n - star)**2, axis=-1)


def chain_problem(d=16, L=1.0, R=1.0):
    D = np.zeros((d - 1, d))
    D[np.arange(d - 1), np.arange(d - 1)] = 1
    D[np.arange(d - 1), np.arange(1, d)] = -1
    star = (d - 1) / 2 - np.arange(d, dtype=float)
    star *= R / np.linalg.norm(star)
    tau = float(np.linalg.norm(D @ star) / math.sqrt(d - 1))
    coefficient = 8 * L / (9 * np.linalg.eigvalsh(D.T @ D)[-1])
    problem = {"D": D, "w_star": star, "tau": tau, "coefficient": coefficient,
               "L": L, "R": R, "dimension": d}
    initial, gradient = chain_value_gradient(np.zeros(d), problem)
    problem.update(initial_gap=float(initial), noise_direction=gradient / np.linalg.norm(gradient))
    return problem


def chain_value_gradient(w, problem, with_value=True):
    """f(w)=8L*tau^2/(9M) sum psi(D(w-w*)/tau), psi(t)=(t^2-log(1+t^2))/2"""
    delta = w - problem["w_star"]
    t = (delta[..., :-1] - delta[..., 1:]) / problem["tau"]
    u = t * t
    value = None
    if with_value:
        small = u < 1e-3
        psi = np.empty_like(u)
        # The series avoids cancellation near the minimum, where psi(t) ~ t^4/4.
        psi[small] = u[small]**2 * (.25 + u[small] * (-1/6 + u[small] *
                       (1/8 + u[small] * (-1/10 + u[small]/12))))
        psi[~small] = 0.5 * (u[~small] - np.log1p(u[~small]))
        value = problem["coefficient"] * problem["tau"]**2 * np.sum(psi, axis=-1)
    edge_gradient = problem["coefficient"] * problem["tau"] * t * (u / (1 + u))
    gradient = np.zeros_like(delta)
    gradient[..., :-1] += edge_gradient
    gradient[..., 1:] -= edge_gradient
    return value, gradient


def simulate_chain(n, runs, problem, noise, method, parameters, seed):
    if method not in ("sgd", "sstm") or (noise.sigma0 > 0 and noise.sigma1 > 0):
        raise ValueError("Use SGD or SSTM with pure additive or multiplicative noise.")
    count = 1 if noise.sigma0 == noise.sigma1 == 0 else runs
    x = np.zeros((count, problem["dimension"]))
    total = np.zeros_like(x)
    y, z, A = x.copy(), x.copy(), 0.0
    rng = np.random.default_rng(seed)
    for k in range(n):
        if method == "sgd":
            total += x
            point = x
            step, lam = parameters["gamma"], parameters["lambda"]
        else:
            step = (k + 2) / (2 * parameters["a"] * problem["L"])
            A_new = A + step
            point = (A * y + step * z) / A_new
            lam = problem["R"] / (30 * step * parameters["H"])
        gradient = chain_value_gradient(point, problem, with_value=False)[1]
        if noise.sigma0 or noise.sigma1:
            innovation = pareto_noise(rng, count, noise.alpha)
            if noise.sigma0:
                gradient += noise.sigma0 * innovation[:, None] * problem["noise_direction"]
            else:
                gradient *= 1 + noise.sigma1 * innovation[:, None]
        norms = np.linalg.norm(gradient, axis=-1)
        factors = np.minimum(1.0, np.divide(lam, norms,
                                          out=np.ones_like(norms), where=norms > 0))
        clipped = gradient * factors[:, None]
        if method == "sgd":
            x -= step * clipped
        else:
            z -= step * clipped
            y = (A * y + step * z) / A_new
            A = A_new
    output = total / n if method == "sgd" else y
    gaps = chain_value_gradient(output, problem)[0]
    return np.full(runs, gaps[0]) if count == 1 else gaps


def log_grid(start, stop, points):
    return np.unique(np.rint(np.geomspace(min(start, stop), stop, points)).astype(np.int64))


def mean_confidence_interval(samples):
    scale = np.max(samples, axis=-1)
    normalized = np.divide(samples, scale[..., None], out=np.zeros_like(samples),
                           where=scale[..., None] > 0)
    mean = normalized.mean(axis=-1) * scale
    se = normalized.std(axis=-1, ddof=1) / math.sqrt(samples.shape[-1]) * scale
    width = student_t.ppf(0.975, samples.shape[-1] - 1) * se
    return mean, np.maximum(0.0, mean - width), mean + width, se


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save_results(directory, number, n, labels, samples, metadata):
    if samples.shape != (len(labels), len(n), metadata["runs"]):
        raise ValueError("Unexpected result shape.")
    if not np.isfinite(samples).all() or np.any(samples < 0):
        raise FloatingPointError("Invalid objective gaps.")
    name = f"experiment{number}"
    metadata = finite_json(metadata)
    np.savez_compressed(directory / f"{name}_runs.npz", n=n, labels=np.asarray(labels),
                        objective_gap=samples, metadata_json=np.asarray(json.dumps(metadata)))
    (directory / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    mean, low, high, se = mean_confidence_interval(samples)
    with (directory / f"{name}_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "oracle_calls", "mean", "ci95_low", "ci95_high",
                         "standard_error", "runs"])
        for c, label in enumerate(labels):
            for j, calls in enumerate(n):
                writer.writerow([label, int(calls), mean[c, j], low[c, j], high[c, j],
                                 se[c, j], samples.shape[-1]])


def configure_plot_style(font_size):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": .9 * font_size, "ytick.labelsize": .9 * font_size,
        "legend.fontsize": .85 * font_size, "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelpad": 3, "legend.handlelength": 2,
        "legend.labelspacing": .25, "legend.borderpad": .35,
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
        "savefig.bbox": None,
    })


def new_axes(args):
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), layout="constrained")
    fig.get_layout_engine().set(w_pad=.07, h_pad=.05)
    ax.set(xscale="log", yscale="log", xlabel=r"Oracle calls $n$")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=.18)
    return fig, ax


def save_figure(fig, directory, name, dpi):
    for extension in ("pdf", "png"):
        temporary = directory / f"{name}.tmp.{extension}"
        fig.savefig(temporary, dpi=dpi, bbox_inches=None)
        temporary.replace(directory / f"{name}.{extension}")
    plt.close(fig)


def plot_average(directory, number, n, labels, samples, metadata, args):
    fig, ax = new_axes(args)
    mean, low, high, _ = mean_confidence_interval(samples)
    bottom = args.ymin or max(1e-18, float(mean[:, -1].min()) * .15)
    top = metadata["initial_gap"] * 1.2 if number == 4 else .7
    ax.set_ylim(bottom, max(top, float(high.max()) * 1.2, bottom * 10))
    for c, label in enumerate(labels):
        color = COLORS[c % len(COLORS)]
        label = label.replace("\\\\", "\\").replace(", common steps", "")
        ax.plot(n, np.where(mean[c] > 0, mean[c], np.nan), color=color,
                linestyle=STYLES[c % len(STYLES)], linewidth=1.2, label=label, zorder=3)
        ax.fill_between(n, np.maximum(low[c], bottom), np.maximum(high[c], bottom),
                        color=color, alpha=.30, linewidth=0, zorder=1)
        for endpoint in (low[c], high[c]):
            ax.plot(n, np.where(endpoint > 0, endpoint, np.nan), color=color,
                    linewidth=.45, alpha=.65, zorder=2)
    for c, root in enumerate(metadata.get("mean_field_roots", [])):
        if root is not None and root != 0:
            ax.axhline(.5 * root**2, color=COLORS[c % len(COLORS)], linestyle=":",
                       linewidth=.8, alpha=.85)
    ax.set_ylabel(r"$f(\bar{w}^{n-1})-f(w^*)$" if number == 3
                  else r"$f(\bar{x}^{n-1})-f^*$")
    ax.legend(loc="best", framealpha=.92)
    save_figure(fig, directory, f"experiment{number}", args.dpi)


def sigma_label(sigma, index):
    if sigma == 0:
        return "No noise"
    exponent = math.floor(math.log10(sigma))
    mantissa = sigma / 10**exponent
    value = (rf"10^{{{exponent}}}" if math.isclose(mantissa, 1)
             else rf"{mantissa:g}\times10^{{{exponent}}}")
    return rf"$\sigma_{index}={value}$"


def plot_sstm(directory, n, samples, metadata, args):
    cases = metadata["oracle_cases"]
    mean, low, high, _ = mean_confidence_interval(samples)
    figures = []
    for index, family in enumerate(("additive", "multiplicative")):
        key, other = f"sigma{index}", f"sigma{1 - index}"
        selected = sorted((i for i, case in enumerate(cases) if case[other] == 0),
                          key=lambda i: cases[i][key])
        fig, ax = new_axes(args)
        for c, i in enumerate(selected):
            sigma = cases[i][key]
            color = COLORS[c % len(COLORS)]
            valid = (low[i] > 0) & np.isfinite(high[i])
            ax.fill_between(n, np.where(valid, low[i], np.nan),
                            np.where(valid, high[i], np.nan), color=color, alpha=.18, linewidth=0)
            marks = sorted(set(range(c % 3, len(n), 3)) | {len(n) - 1})
            ax.plot(n, np.where(mean[i] > 0, mean[i], np.nan), color=color,
                    label=sigma_label(sigma, index), linestyle=":" if sigma == 0 else STYLES[c % len(STYLES)],
                    linewidth=1.1 if sigma == 0 else 1.4, marker="o" if sigma == 0 else "s",
                    markersize=3.4, markerfacecolor="white", markeredgewidth=.8,
                    markevery=marks, zorder=20 + len(selected) if sigma == 0 else 3 + c)
        ax.set(title=rf"{family.capitalize()} noise ($\sigma_{1-index}=0$)",
               ylabel=r"Mean objective gap $f(y^n)-f^*$")
        ax.legend(loc="best", framealpha=.92)
        figures.append((fig, ax, family))
    limits = (min(ax.get_ylim()[0] for _, ax, _ in figures),
              max(ax.get_ylim()[1] for _, ax, _ in figures))
    for fig, ax, family in figures:
        ax.set_ylim(limits)
        save_figure(fig, directory, f"experiment5_{family}", args.dpi)


def experiment1(args):
    n = log_grid(1, args.n1, args.points)
    labels = [rf"$\sigma_0={s:g}$" for s in SIGMA0_LEVELS]
    samples = simulate_quadratic(args.n1, args.runs, n, SIGMA0_LEVELS, .03,
                                args.alpha, .05, [1.0], [args.seed, 1], symmetric=True)
    metadata = {"objective": "x^2/2", "initial_gap": .5, "sigma0": list(SIGMA0_LEVELS),
                "sigma1": .03, "gamma": .05, "lambda": 1.0, "noise": "symmetric",
                "protocol": "single trajectory with checkpoints", "seed_sequence": "[seed,1]"}
    return n, labels, samples, metadata


def experiment2(args):
    noise = Noise(alpha=args.alpha)
    n = log_grid(args.n2_min, args.n2, args.horizons)
    labels = [rf"Fixed $\lambda={lam:g}$" for lam in CLIPPING_LEVELS] + [r"Theorem 5.1 $\lambda(n)$"]
    samples = np.empty((len(labels), len(n), args.runs))
    settings = []
    for j, calls in enumerate(n):
        parameters = sgd_parameters(int(calls), noise, args.beta)
        lambdas = [*CLIPPING_LEVELS, parameters["lambda"]]
        print(f"  n={calls:,}, reference lambda={lambdas[-1]:.6g}", flush=True)
        samples[:, j] = simulate_quadratic(int(calls), args.runs, [calls], [noise.sigma0],
                                          noise.sigma1, noise.alpha, args.gamma2, lambdas,
                                          [args.seed, 2, int(calls)])[:, 0]
        settings.append(parameters | {"gamma_actual": args.gamma2, "lambda_actual": lambdas})
    metadata = {"objective": "x^2/2", "initial_gap": .5, "noise": "centered",
                "noise_parameters": asdict(noise), "gamma": args.gamma2,
                "fixed_lambdas": list(CLIPPING_LEVELS), "settings": settings,
                "mean_field_roots": [mean_field_root(lam, noise) for lam in CLIPPING_LEVELS],
                "protocol": "restart per horizon; common step; paired innovations across thresholds",
                "seed_sequence": "[seed,2,horizon]"}
    return n, labels, samples, metadata


def experiment3(args):
    problem = regression_problem(args.rho3)
    noise = Noise(args.alpha, args.sigma03, args.sigma13)
    n = log_grid(args.n2_min, args.n3, args.horizons)
    gamma = args.gamma3 if args.gamma3 is not None else .05 / problem["L"]
    if gamma > 1 / problem["L"]:
        raise ValueError("Experiment 3 requires gamma3 <= 1/L.")
    labels = [rf"Fixed $\lambda={lam:g}$" for lam in CLIPPING_LEVELS] + [r"Theorem 6 $\lambda(n)$"]
    samples = np.empty((len(labels), len(n), args.runs))
    settings = []
    for j, calls in enumerate(n):
        parameters = sgd_parameters(int(calls), noise, args.beta, problem["L"], problem["R"])
        lambdas = [*CLIPPING_LEVELS, parameters["lambda"]]
        print(f"  n={calls:,}, reference lambda={lambdas[-1]:.6g}", flush=True)
        samples[:, j] = simulate_regression(int(calls), args.runs, problem, noise, gamma,
                                           lambdas, [args.seed, 3, int(calls)], args.scale3)
        settings.append(parameters | {"gamma_actual": gamma, "lambda_actual": lambdas})
    np.savez_compressed(args.out / "experiment3_problem.npz", **{k: problem[k] for k in
                        ("A", "y", "H", "w_star", "eigenvalues", "eigenvectors",
                         "X_mean", "X_std", "y_mean", "y_std")})
    metadata = {"objective": "||A w-y||^2/(2m)+rho*||w||^2/2", "rho": args.rho3,
                "L": problem["L"], "R": problem["R"], "gamma": gamma,
                "initial_gap": float(.5 * np.sum(problem["eigenvalues"] * problem["q_star"]**2)),
                "dataset": "Diabetes", "noise": "centered",
                "preprocessing": "standardize all features and targets with population standard deviations",
                "noise_parameters": asdict(noise), "scale_model": args.scale3,
                "direction": "smallest Hessian eigenvalue; largest absolute eigenvector entry positive",
                "settings": settings, "fixed_lambdas": list(CLIPPING_LEVELS),
                "protocol": "restart per horizon; common step; paired innovations across thresholds",
                "seed_sequence": "[seed,3,horizon]"}
    return n, labels, samples, metadata


def chain_metadata(problem):
    return {"objective": "8L*tau^2/(9M)*sum psi(D(w-w*)/tau); psi(t)=(t^2-log(1+t^2))/2",
            "dimension": problem["dimension"], "L": problem["L"], "R": problem["R"],
            "tau": problem["tau"], "initial_gap": problem["initial_gap"],
            "protocol": "independent restart for each noise case and horizon", "noise": "centered"}


def experiment4(args):
    problem = chain_problem(args.chain_dim)
    n = log_grid(args.n4_min, args.n4, args.rates_horizons)
    pairs = [(0., 0.), (0., args.rates_sigma1), (args.rates_sigma0, 0.)]
    labels = ["No noise", "Multiplicative noise", "Additive noise"]
    samples = np.empty((len(pairs), len(n), args.runs))
    settings = []
    for c, (s0, s1) in enumerate(pairs):
        noise = Noise(args.alpha, s0, s1)
        case_settings = []
        for j, calls in enumerate(n):
            parameters = sgd_parameters(int(calls), noise, args.beta)
            print(f"  sigma=({s0:g},{s1:g}), n={calls:,}, gamma={parameters['gamma']:.6g}", flush=True)
            samples[c, j] = simulate_chain(int(calls), args.runs, problem, noise, "sgd",
                                          parameters, [args.seed, 4, c, int(calls)])
            case_settings.append(parameters)
        settings.append(case_settings)
    np.savez_compressed(args.out / "experiment4_problem.npz", D=problem["D"],
                        w_star=problem["w_star"], noise_direction=problem["noise_direction"])
    metadata = chain_metadata(problem) | {"oracle_cases": pairs, "settings": settings,
                                          "seed_sequence": "[seed,4,case,horizon]"}
    return n, labels, samples, metadata


def experiment5(args):
    problem = chain_problem(args.chain_dim)
    n = log_grid(args.n5_min, args.n5, args.sstm_horizons)
    pairs = [(0., 0.)] + [(s, 0.) for s in args.sstm_sigma0 if s > 0]
    pairs += [(0., s) for s in args.sstm_sigma1 if s > 0]
    labels = ["No noise" if s0 == s1 == 0 else f"sigma0={s0:g}, sigma1={s1:g}" for s0, s1 in pairs]
    samples = np.empty((len(pairs), len(n), args.runs))
    settings = []
    for c, (s0, s1) in enumerate(pairs):
        noise = Noise(args.alpha, s0, s1)
        case_settings = []
        for j, calls in enumerate(n):
            parameters = sstm_parameters(int(calls), noise, args.beta)
            print(f"  sigma=({s0:g},{s1:g}), n={calls:,}, a={parameters['a']:.6g}", flush=True)
            samples[c, j] = simulate_chain(int(calls), args.runs, problem, noise, "sstm",
                                          parameters, [args.seed, 5, c, int(calls)])
            case_settings.append(parameters)
        settings.append(case_settings)
    np.savez_compressed(args.out / "experiment5_problem.npz", D=problem["D"],
                        w_star=problem["w_star"], noise_direction=problem["noise_direction"])
    metadata = chain_metadata(problem) | {
        "oracle_cases": [{"sigma0": s0, "sigma1": s1} for s0, s1 in pairs],
        "settings": settings, "seed_sequence": "[seed,5,case,horizon]"}
    return n, labels, samples, metadata


def load_results(directory, number):
    path = directory / f"experiment{number}_runs.npz"
    with np.load(path, allow_pickle=False) as data:
        key = "objective_gap" if "objective_gap" in data else "averaged_objective"
        n, labels, samples = data["n"].copy(), data["labels"].tolist(), data[key].copy()
        metadata = json.loads(str(data["metadata_json"].item()))
    if samples.ndim != 3 or samples.shape[:2] != (len(labels), len(n)):
        raise ValueError(f"Invalid sample dimensions in {path}")
    if samples.shape[-1] < 2 or not np.isfinite(samples).all() or np.any(samples < 0):
        raise ValueError(f"Invalid objective samples in {path}")
    if number == 5 and any(case.get("noise", "centered") not in ("none", "centered")
                           for case in metadata["oracle_cases"]):
        raise ValueError("Experiment 5 expects the centered Pareto noise used in the article.")
    return n, labels, samples, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "1", "2", "3", "4", "5", "rates", "sstm"), default="all")
    parser.add_argument("--n", type=int, default=1_000_000, help="Maximum calls; --n1,...,--n5 override it.")
    for i in range(1, 6):
        parser.add_argument(f"--n{i}", type=int)
    parser.add_argument("--runs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=.05)
    parser.add_argument("--points", type=int, default=180, help="Checkpoints in experiment 1.")
    parser.add_argument("--horizons", type=int, default=24, help="Restarted horizons in experiments 2 and 3.")
    parser.add_argument("--n2-min", type=int, default=64)
    parser.add_argument("--gamma2", type=float, default=.05)
    parser.add_argument("--rho3", type=float, default=.01)
    parser.add_argument("--sigma03", type=float, default=.05)
    parser.add_argument("--sigma13", type=float, default=.001)
    parser.add_argument("--gamma3", type=float)
    parser.add_argument("--scale3", choices=("projected", "full"), default="projected")
    parser.add_argument("--chain-dim", type=int, default=16)
    parser.add_argument("--n4-min", type=int, default=1000)
    parser.add_argument("--n5-min", type=int, default=1000)
    parser.add_argument("--rates-horizons", type=int, default=8)
    parser.add_argument("--sstm-horizons", type=int, default=8)
    parser.add_argument("--rates-sigma0", type=float, default=.002)
    parser.add_argument("--rates-sigma1", type=float, default=.004)
    parser.add_argument("--sstm-sigma0", type=float, nargs="+", default=list(SSTM_SIGMA0))
    parser.add_argument("--sstm-sigma1", type=float, nargs="+", default=list(SSTM_SIGMA1))
    parser.add_argument("--fig-width", type=float, default=3.25)
    parser.add_argument("--fig-height", type=float, default=2.65)
    parser.add_argument("--font-size", type=float, default=10)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--replot", type=Path, help="Redraw saved NPZ results without simulation.")
    args = parser.parse_args()
    for i in range(1, 6):
        if getattr(args, f"n{i}") is None:
            setattr(args, f"n{i}", args.n)
    integers = [args.n, args.n1, args.n2, args.n3, args.n4, args.n5,
                args.n2_min, args.n4_min, args.n5_min, args.dpi]
    if min(integers) < 1 or args.runs < 2 or args.seed < 0 or args.chain_dim < 2:
        parser.error("Positive horizons and dpi, runs>=2, seed>=0 and chain-dim>=2 are required.")
    if min(args.points, args.horizons, args.rates_horizons, args.sstm_horizons) < 2:
        parser.error("Use at least two grid points.")
    if not 1 < args.alpha <= 2 or not 0 < args.beta < 1:
        parser.error("Require 1<alpha<=2 and 0<beta<1.")
    positive = [args.gamma2, args.rho3, args.fig_width, args.fig_height, args.font_size]
    positive += [v for v in (args.gamma3, args.ymin) if v is not None]
    if not all(math.isfinite(v) and v > 0 for v in positive) or args.gamma2 > 1:
        parser.error("Steps, regularization and figure dimensions must be positive; gamma2<=1.")
    levels = [args.sigma03, args.sigma13, args.rates_sigma0, args.rates_sigma1,
              *args.sstm_sigma0, *args.sstm_sigma1]
    if not all(math.isfinite(v) and v >= 0 for v in levels):
        parser.error("Noise coefficients must be finite and nonnegative.")
    if len(set(args.sstm_sigma0)) != len(args.sstm_sigma0) or len(set(args.sstm_sigma1)) != len(args.sstm_sigma1):
        parser.error("Repeated SSTM noise levels are not allowed.")
    return args


def main():
    args = parse_args()
    configure_plot_style(args.font_size)
    args.out.mkdir(parents=True, exist_ok=True)
    selection = {"all": (1, 2, 3, 4, 5), "rates": (4,), "sstm": (5,)}
    numbers = selection[args.experiment] if args.experiment in selection else (int(args.experiment),)
    experiments = {1: experiment1, 2: experiment2, 3: experiment3, 4: experiment4, 5: experiment5}
    started = time.perf_counter()
    for number in numbers:
        print(f"Experiment {number}", flush=True)
        if args.replot:
            n, labels, samples, metadata = load_results(args.replot, number)
        else:
            n, labels, samples, metadata = experiments[number](args)
            metadata.update(experiment=number, runs=args.runs, seed=args.seed, alpha=args.alpha,
                            beta_theorem=args.beta, horizons=n.tolist(),
                            output="y^n" if number == 5 else "preupdate average",
                            confidence_interval="pointwise approximate Student-t 95% CI for the mean")
            save_results(args.out, number, n, labels, samples, metadata)
        if number == 5:
            plot_sstm(args.out, n, samples, metadata, args)
        else:
            plot_average(args.out, number, n, labels, samples, metadata, args)
    print(f"Saved to {args.out.resolve()} ({time.perf_counter() - started:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
