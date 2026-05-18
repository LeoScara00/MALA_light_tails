def plot_experiment(result, p, h, lyapunov_label, eps, output_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it first, then rerun with --plot."
        ) from exc

    x_grid = result["x_grid"]
    region_names = tuple(result["regional_drifts"].keys())
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

    for name in region_names:
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

    for name in region_names:
        axes[2].plot(x_grid, result["proposal_probs"][name], linewidth=2, label=name)
    axes[2].set_ylabel("Q(x, A_k)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(frameon=False)

    for name in region_names:
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
