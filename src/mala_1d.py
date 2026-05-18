import argparse

from src.diagnostics import (
    alpha,
    compute_R,
    delta,
    estimate_I,
    estimate_drift_statistics,
    print_experiment_summary,
    run_experiment,
    sample_proposal,
)
from src.lyapunov import (
    U,
    V,
    V_U,
    V_exp,
    V_poly,
    build_lyapunov,
    dU,
    evaluate_lyapunov_difference,
    log_V_U,
    log_V_exp,
)
from src.partitions import REGION_NAMES, classify_regions
from src.plotting import plot_experiment
from src.utils import MAX_LOG_VALUE, mean_and_se, parse_x_grid, subtract_from_logs


__all__ = [
    "MAX_LOG_VALUE",
    "REGION_NAMES",
    "U",
    "V",
    "V_U",
    "V_exp",
    "V_poly",
    "alpha",
    "build_lyapunov",
    "classify_regions",
    "compute_R",
    "dU",
    "delta",
    "estimate_I",
    "estimate_drift_statistics",
    "evaluate_lyapunov_difference",
    "log_V_U",
    "log_V_exp",
    "mean_and_se",
    "parse_x_grid",
    "plot_experiment",
    "print_experiment_summary",
    "run_experiment",
    "sample_proposal",
    "subtract_from_logs",
]


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
        default="results/figures/mala_1d_diagnostics.png",
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
