import warnings

import numpy as np

from src.lyapunov import V_poly, U, dU, evaluate_lyapunov_difference
from src.partitions import REGION_NAMES, classify_regions
from src.utils import mean_and_se


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
