import argparse
import warnings

import numpy as np


MAX_LOG_VALUE = 700.0
REGION_NAMES = ("A1", "A2", "A3")


def U(x, p):
    return np.abs(x) ** (2 * p)


def dU(x, p):
    return 2 * p * x * np.abs(x) ** (2 * p - 2)


def V_poly(x, m):
    return 1.0 + np.abs(x) ** m


def V(x, m):
    return V_poly(x, m)


def log_V_exp(x, s, beta):
    return s * np.abs(x) ** beta


def V_exp(x, s, beta):
    return np.exp(np.clip(log_V_exp(x, s, beta), a_min=None, a_max=MAX_LOG_VALUE))


def log_V_U(x, s, q, p):
    return s * U(x, p) ** q


def V_U(x, s, q, p):
    return np.exp(np.clip(log_V_U(x, s, q, p), a_min=None, a_max=MAX_LOG_VALUE))


def sample_proposal(x, h, p, rng, size=None):
    epsilon = rng.normal(size=size)
    drift = 0.5 * h**2 * dU(x, p)
    return x - drift + h * epsilon


def delta(x, y, h, p):
    grad_x = dU(x, p)
    grad_y = dU(y, p)

    potential_change = U(y, p) - U(x, p)
    trapezoid_term = 0.5 * (x - y) * (grad_x + grad_y)
    gradient_term = 0.125 * h**2 * (grad_y**2 - grad_x**2)
    return potential_change + trapezoid_term + gradient_term


def alpha(x, y, h, p):
    delta_xy = delta(x, y, h, p)
    return np.exp(np.minimum(-delta_xy, 0.0))


def subtract_from_logs(log_a, log_b):
    log_a, log_b = np.broadcast_arrays(
        np.asarray(log_a, dtype=float),
        np.asarray(log_b, dtype=float),
    )
    result = np.empty(log_a.shape, dtype=float)
    mask = log_a >= log_b

    result[mask] = np.exp(
        np.clip(log_a[mask], a_min=None, a_max=MAX_LOG_VALUE)
    ) * (-np.expm1(log_b[mask] - log_a[mask]))
    result[~mask] = -np.exp(
        np.clip(log_b[~mask], a_min=None, a_max=MAX_LOG_VALUE)
    ) * (-np.expm1(log_a[~mask] - log_b[~mask]))
    return result


def build_lyapunov(lyapunov_type, p, m=2.0, s=0.1, beta=1.0, q=1.0):
    if lyapunov_type == "poly":
        return {
            "name": "poly",
            "label": f"V(x)=1+|x|^{m}",
            "V_func": lambda x: V_poly(x, m),
            "log_V_func": None,
            "params": {"m": m},
        }

    if lyapunov_type == "exp":
        return {
            "name": "exp",
            "label": f"V(x)=exp({s}|x|^{beta})",
            "V_func": lambda x: V_exp(x, s, beta),
            "log_V_func": lambda x: log_V_exp(x, s, beta),
            "params": {"s": s, "beta": beta},
        }

    if lyapunov_type == "U":
        return {
            "name": "U",
            "label": f"V(x)=exp({s} U(x)^{q})",
            "V_func": lambda x: V_U(x, s, q, p),
            "log_V_func": lambda x: log_V_U(x, s, q, p),
            "params": {"s": s, "q": q, "p": p},
        }

    raise ValueError(f"Unknown Lyapunov type: {lyapunov_type}")


def evaluate_lyapunov_difference(y, x, V_func, log_V_func=None):
    if log_V_func is not None:
        return subtract_from_logs(log_V_func(y), log_V_func(x))
    return V_func(y) - V_func(x)


def classify_regions(x, y, eps):
    if x <= 0:
        raise ValueError("Region diagnostics require x > 0.")

    left = (1.0 - eps) * x
    right = (1.0 + eps) * x

    return {
        "A1": y <= left,
        "A2": (y > left) & (y < right),
        "A3": y >= right,
    }


def mean_and_se(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if values.size <= 1:
        return mean, 0.0
    se = values.std(ddof=1) / np.sqrt(values.size)
    return mean, se


def estimate_drift_statistics(x, h, p, n_samples, rng, V_func, log_V_func=None, eps=0.1):
    y = sample_proposal(x, h, p, rng, size=n_samples)
    acceptances = alpha(x, y, h, p)
    drift_terms = evaluate_lyapunov_difference(y, x, V_func, log_V_func) * acceptances

    total_drift, total_se = mean_and_se(drift_terms)
    regions = classify_regions(x, y, eps)

    regional_drifts = {}
    proposal_probs = {}
    accepted_probs = {}

    for name in REGION_NAMES:
        mask = regions[name]
        masked_drift = drift_terms * mask
        masked_acceptance = acceptances * mask

        drift_mean, drift_se = mean_and_se(masked_drift)
        regional_drifts[name] = {"mean": drift_mean, "se": drift_se}
        proposal_probs[name] = mask.mean()
        accepted_probs[name] = masked_acceptance.mean()

    return {
        "x": x,
        "total_drift": total_drift,
        "total_se": total_se,
        "mean_acceptance": acceptances.mean(),
        "regional_drifts": regional_drifts,
        "proposal_probs": proposal_probs,
        "accepted_probs": accepted_probs,
    }


def estimate_I(x, h, p, m=None, n_samples=100_000, rng=None, V_func=None, log_V_func=None):
    if rng is None:
        rng = np.random.default_rng()

    if V_func is None:
        if m is None:
            raise ValueError("Provide either m for the polynomial Lyapunov function or V_func.")
        V_func = lambda z: V_poly(z, m)

    stats = estimate_drift_statistics(
        x=x,
        h=h,
        p=p,
        n_samples=n_samples,
        rng=rng,
        V_func=V_func,
        log_V_func=log_V_func,
        eps=0.1,
    )
    return stats["total_drift"], stats["mean_acceptance"]


def compute_R(x, h, p):
    x = np.asarray(x, dtype=float)
    numerator = h**2 * np.abs(dU(x, p))
    denominator = np.abs(x)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def parse_x_grid(text):
    if ":" in text:
        parts = [part.strip() for part in text.split(":") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                "Range-style x-grid must have the form 'start:stop:num', for example '0.5:6:30'."
            )
        start, stop, num = float(parts[0]), float(parts[1]), int(float(parts[2]))
        return np.linspace(start, stop, num)

    parts = [part.strip() for part in text.split(",") if part.strip()]
    return np.array([float(part) for part in parts], dtype=float)


def run_experiment(x_grid, h, p, n_samples, seed, V_func, log_V_func=None, eps=0.1):
    if np.any(x_grid <= 0):
        raise ValueError("This regional diagnostic setup assumes x_grid > 0.")

    rng = np.random.default_rng(seed)
    result = {
        "x_grid": np.asarray(x_grid, dtype=float),
        "total_drift": np.empty(len(x_grid)),
        "total_se": np.empty(len(x_grid)),
        "mean_acceptance": np.empty(len(x_grid)),
        "R_values": compute_R(x_grid, h, p),
        "regional_drifts": {name: np.empty(len(x_grid)) for name in REGION_NAMES},
        "regional_se": {name: np.empty(len(x_grid)) for name in REGION_NAMES},
        "proposal_probs": {name: np.empty(len(x_grid)) for name in REGION_NAMES},
        "accepted_probs": {name: np.empty(len(x_grid)) for name in REGION_NAMES},
    }

    for i, x in enumerate(x_grid):
        stats = estimate_drift_statistics(
            x=x,
            h=h,
            p=p,
            n_samples=n_samples,
            rng=rng,
            V_func=V_func,
            log_V_func=log_V_func,
            eps=eps,
        )

        result["total_drift"][i] = stats["total_drift"]
        result["total_se"][i] = stats["total_se"]
        result["mean_acceptance"][i] = stats["mean_acceptance"]

        for name in REGION_NAMES:
            result["regional_drifts"][name][i] = stats["regional_drifts"][name]["mean"]
            result["regional_se"][name][i] = stats["regional_drifts"][name]["se"]
            result["proposal_probs"][name][i] = stats["proposal_probs"][name]
            result["accepted_probs"][name][i] = stats["accepted_probs"][name]

    if result["R_values"][-1] > 1.0:
        warnings.warn(
            f"R(x_max) = {result['R_values'][-1]:.4f} > 1. "
            "The proposal drift may be too large at the edge of the grid.",
            RuntimeWarning,
        )

    return result


def print_experiment_summary(result, p, h, lyapunov_label, eps):
    print("1D MALA drift diagnostics")
    print(f"p={p}, h={h}, eps={eps}, Lyapunov={lyapunov_label}")
    print()
    print("x        drift        se          acc       Q(A1)     Q(A2)     Q(A3)")

    for i, x in enumerate(result["x_grid"]):
        print(
            f"{x:6.3f}   "
            f"{result['total_drift'][i]: .6e}   "
            f"{result['total_se'][i]: .3e}   "
            f"{result['mean_acceptance'][i]: .6f}   "
            f"{result['proposal_probs']['A1'][i]: .6f}   "
            f"{result['proposal_probs']['A2'][i]: .6f}   "
            f"{result['proposal_probs']['A3'][i]: .6f}"
        )

    print()
    print(f"R(x_max) = {result['R_values'][-1]:.6f}")


def plot_experiment(result, p, h, lyapunov_label, eps, output_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it first, then rerun with --plot."
        ) from exc

    x_grid = result["x_grid"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 18), sharex=True)

    axes[0].plot(x_grid, result["total_drift"], color="black", linewidth=2, label="total drift")
    axes[0].fill_between(
        x_grid,
        result["total_drift"] - result["total_se"],
        result["total_drift"] + result["total_se"],
        color="black",
        alpha=0.15,
    )
    axes[0].set_ylabel("Total drift")
    axes[0].set_title(f"1D MALA drift diagnostics: p={p}, h={h}, eps={eps}\n{lyapunov_label}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    for name in REGION_NAMES:
        axes[1].plot(x_grid, result["regional_drifts"][name], linewidth=2, label=name)
        axes[1].fill_between(
            x_grid,
            result["regional_drifts"][name] - result["regional_se"][name],
            result["regional_drifts"][name] + result["regional_se"][name],
            alpha=0.15,
        )
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].set_ylabel("Regional drift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    for name in REGION_NAMES:
        axes[2].plot(x_grid, result["proposal_probs"][name], linewidth=2, label=name)
    axes[2].set_ylabel("Q(x, A_k)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(frameon=False)

    for name in REGION_NAMES:
        axes[3].plot(x_grid, result["accepted_probs"][name], linewidth=2, label=name)
    axes[3].plot(
        x_grid,
        result["mean_acceptance"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="total acceptance",
    )
    axes[3].set_ylabel("E[alpha 1_Ak]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(frameon=False)

    axes[4].plot(x_grid, result["R_values"], color="tab:red", linewidth=2)
    axes[4].axhline(1.0, color="gray", linewidth=1, linestyle="--")
    axes[4].set_xlabel("x")
    axes[4].set_ylabel("R(x)")
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run 1D MALA drift diagnostics over an x-grid."
    )
    parser.add_argument("--p", type=int, default=2, help="Parameter p in U(x)=|x|^(2p).")
    parser.add_argument("--h", type=float, default=0.05, help="MALA step size.")
    parser.add_argument("--samples", type=int, default=100_000, help="Monte Carlo samples per x.")
    parser.add_argument("--eps", type=float, default=0.1, help="Region-width parameter.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--x-grid",
        type=str,
        default="0.5:6.0:30",
        help="Grid of x values, either 'a:b:n' or an explicit comma-separated list.",
    )
    parser.add_argument(
        "--lyapunov",
        choices=["poly", "exp", "U"],
        default="poly",
        help="Choice of Lyapunov function.",
    )
    parser.add_argument("--m", type=float, default=1.0, help="Exponent m for V_poly.")
    parser.add_argument("--s", type=float, default=0.1, help="Scale s for exponential Lyapunov functions.")
    parser.add_argument("--beta", type=float, default=1.0, help="Exponent beta for V_exp.")
    parser.add_argument("--q", type=float, default=1.0, help="Exponent q for V_U.")
    parser.add_argument("--plot", action="store_true", help="Plot all diagnostics.")
    parser.add_argument(
        "--output",
        type=str,
        default="mala_1d_diagnostics.png",
        help="Output filename used when --plot is passed.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    x_grid = parse_x_grid(args.x_grid)
    lyapunov = build_lyapunov(
        lyapunov_type=args.lyapunov,
        p=args.p,
        m=args.m,
        s=args.s,
        beta=args.beta,
        q=args.q,
    )

    result = run_experiment(
        x_grid=x_grid,
        h=args.h,
        p=args.p,
        n_samples=args.samples,
        seed=args.seed,
        V_func=lyapunov["V_func"],
        log_V_func=lyapunov["log_V_func"],
        eps=args.eps,
    )

    print_experiment_summary(
        result=result,
        p=args.p,
        h=args.h,
        lyapunov_label=lyapunov["label"],
        eps=args.eps,
    )

    if args.plot:
        plot_experiment(
            result=result,
            p=args.p,
            h=args.h,
            lyapunov_label=lyapunov["label"],
            eps=args.eps,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
