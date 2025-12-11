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

n_sims=16
test_sizes = [1024]
np.logspace(np.log10(100), np.log10(10_000), 20, dtype=int)
for n_runs in [8, 16, 32, 64, 128, 256, 512, 1028]:
    print(f"\n***** {n_runs=}")
    ref_size = 100_000
    all_stats = []
    all_stats.extend(compute_stats(ref_size, test_sizes, n_sims, n_runs))
    df = pd.DataFrame(all_stats)
    print(df)

