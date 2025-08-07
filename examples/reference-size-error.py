import numpy as np
import pandas as pd
from plotnine import *

def compute_stats(ref_size, test_sizes, n_samples=1_000_000):
    stats = []
    for test_size in test_sizes:
        std_dev = np.sqrt(1 / ref_size + 1 / test_size)
        samples = np.abs(np.random.normal(0, std_dev, size=n_samples))
        lower = np.percentile(samples, 5)
        upper = np.percentile(samples, 95)
        mean = samples.mean()
        stats.append({
            'test_size': test_size,
            'ref_size': f'{ref_size:,}',
            'mean': mean,
            'lower': lower,
            'upper': upper
        })
    return stats

def compute_stats_old(ref_size, test_sizes, n_samples=1_000_000):
    stats = []
    for test_size in test_sizes:
        std_dev = np.sqrt(1 / ref_size + 1 / test_size)
        samples = np.random.normal(loc=0, scale=std_dev, size=n_samples)
        abs_samples = np.abs(samples)
        mean = abs_samples.mean()
        std_dev = abs_samples.std(ddof=1)
        stats.append({
            'test_size': test_size,
            'ref_size': f'{ref_size:,}',
            'mean': mean,
            'lower': mean - 1.96 * std_dev,
            'upper': mean + 1.96 * std_dev
        })
    print(stats)
    return stats

ref_sizes = [10_000, np.inf]
test_sizes = np.logspace(np.log10(100), np.log10(1_000_000), 50)

all_stats = []
for ref_size in ref_sizes:
    all_stats.extend(compute_stats(ref_size, test_sizes))

df = pd.DataFrame(all_stats)

plot = (
    ggplot(df, aes(x='test_size'))
    + geom_line(aes(y='mean', color='ref_size', fill='ref_size'), size=1)
    + geom_ribbon(aes(ymin='lower', ymax='upper', fill='ref_size', group='ref_size'), alpha=0.2)
    + scale_x_log10()
    + scale_y_log10()
    + labs(
        x='Test Size',
        y='E[|Y|]',
        color='Reference Size',
        fill='Reference Size'
    )
    + theme_minimal()
)
plot.show()
