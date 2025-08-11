import numpy as np
import pandas as pd
from plotnine import *

def compute_stats(ref_size, test_sizes, n_sims, n_runs):
    ref_size_id = f'{ref_size:,}'
    stats = []
    for test_size in test_sizes:
        # print(f"{test_size=}")
        std_dev = np.sqrt(1 / ref_size + 1 / test_size)
        esss = np.zeros(n_sims)
        for m in range(n_sims):
            errs = np.zeros(n_runs)
            for n in range(n_runs):
                sample = np.random.normal(0, std_dev, size=test_size)
                errs[n] = sample.mean()
            esss[m] = 1 / np.std(errs)
        mean = np.mean(esss)
        lower = np.percentile(esss, 5)
        upper = np.percentile(esss, 95)
        stats.append({
            'test_size': test_size,
            'ref_size': ref_size_id,
            'mean': mean,
            'lower': lower,
            'upper': upper
        })
    return stats

n_sims=5000
test_sizes = [1000]
# np.logspace(np.log10(100), np.log10(10_000), 20, dtype=int)
for n_runs in [256, 512, 1028]:
    print(f"\n***** {n_runs=}")
    for ref_size in [10_000, 100_000]:
        ref_sizes = [ref_size] # [ref_size, np.inf]
        
        all_stats = []
        for ref_size in ref_sizes:
            all_stats.extend(compute_stats(ref_size, test_sizes, n_sims, n_runs))

        df = pd.DataFrame(all_stats)
        print(df)

        
#         plot = (
#             ggplot(df, aes(x='test_size'))
#             + geom_line(aes(y='mean', color='ref_size', fill='ref_size'), size=1)
#             + geom_ribbon(aes(ymin='lower', ymax='upper', fill='ref_size', group='ref_size'), alpha=0.2)
#             + scale_x_log10(breaks=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
#     + scale_y_log10()
#     + labs(
#         x='test sample size',
#         y='estimated ESS',
#         title=f"Number of test runs: {n_runs}",
#         color='reference Size',
#         fill='reference Size'
#     )
# )
# plot.show()
