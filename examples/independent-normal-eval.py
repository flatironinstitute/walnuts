import numpy as np
import pandas as pd
import plotnine as pn


def compute_thinned_errors(rng, reference_counts, total_draws, log_indices):
    """Generate draws and return thinned absolute errors for each reference count."""
    ref_draws = rng.standard_normal(1_000_001)
    reference_means = {f"{n:,} draws": np.mean(ref_draws[:n]) for n in reference_counts}
    draws = rng.standard_normal(total_draws)
    cumsum = np.cumsum(draws)
    cummean = cumsum / np.arange(1, total_draws + 1)
    thinned = cummean[log_indices - 1]
    errors = {label: np.abs(thinned - m) for label, m in reference_means.items()}
    errors["true value"] = np.abs(thinned)
    return errors


def main():
    rng = np.random.default_rng(seed=977473)
    reference_counts = [1_000, 10_000, 100_000, 1_000_000]
    log_indices = np.unique(np.logspace(3, 8, num=400, dtype=int))
    total_draws = 100_000_000
    runs = 1

    aggregated = {
        f"{n:,} draws": np.zeros_like(log_indices, dtype=float)
        for n in reference_counts
    }
    aggregated["true value"] = np.zeros_like(log_indices, dtype=float)

    for i in range(runs):
        print(f"Iteration {i+1:4d} / {runs}")
        errs = compute_thinned_errors(rng, reference_counts, total_draws, log_indices)
        for label, vals in errs.items():
            aggregated[label] += vals

    averaged = {label: vals / runs for label, vals in aggregated.items()}
    df = pd.DataFrame({"iteration": log_indices, **averaged})
    df_long = df.melt(id_vars="iteration", var_name="reference", value_name="error")

    plot = (
        pn.ggplot(df_long, pn.aes("iteration", "error", color="reference"))
        + pn.geom_line()
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.labs(x="test sample size", y="absolute error", title="Error vs. Sample Size (by reference size)", color="reference")
        + pn.theme_minimal()
    )
    plot.save("avg_reference_evaluation_plot.jpg", dpi=300, width=9, height=6)
    plot.save("avg_reference_evaluation_plot.pdf", width=9, height=6)
    plot.show()


if __name__ == "__main__":
    main()
