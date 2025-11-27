import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_point, geom_ribbon, geom_hline, labs, theme_minimal, scale_x_log10, scale_y_continuous

def compute_stats(ref_size: int, test_size: int, n_sims: int, n_runs: int) -> dict:
    print(f"{n_runs=}")
    std_dev = np.sqrt(1 / ref_size + 1 / test_size)
    errs = np.empty((n_sims, n_runs))
    se = np.empty(n_sims)
    for m in range(n_sims):
        errs[m] = np.random.normal(0.0, std_dev, size=n_runs)
        se[m] = errs[m].std(ddof=1)
    ess = 1 / se**2
    return {
        "n_runs": n_runs,
        "test_size": test_size,
        "ref_size": f"{ref_size:,}",
        "median": float(np.median(ess)),
        "lower": float(np.percentile(ess, 5)),
        "upper": float(np.percentile(ess, 95)),
    }

n_sims = 10_000
ref_size = 10_000
test_size = 1000
n_runs_list = [16, 32, 64, 128, 256, 512]

rows = [compute_stats(ref_size, test_size, n_sims, n_runs) for n_runs in n_runs_list]
df = pd.DataFrame(rows)

plot = (
    ggplot(df, aes("n_runs", "median"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), alpha=0.20, fill="steelblue")
    + geom_hline(yintercept=1000, linetype="dashed", color="red")
    + geom_line()
    + geom_point()
    + scale_x_log10(breaks=n_runs_list)
    + scale_y_continuous(breaks=range(0, 5000, 200))
    + labs(
        x="sampler runs",
        y="estimated effective sample size",
        title=f"ESS vs runs (test_size={test_size:,};  ref_size={ref_size:,})",
    )
    + theme_minimal()
)
plot.save('ess-power-plot-10K.pdf')
plot.show()
