import numpy as np
import pandas as pd
from plotnine import *

def compute_stats(ref_size, test_sizes, n_samples=1_000_000):
    stats = []
    for test_size in test_sizes:
        std_dev = np.sqrt(1 / ref_size + 1 / test_size)
        samples = np.random.normal(0, std_dev, size=n_samples)
        tr_samples = np.abs(samples)**2
        lower = np.percentile(tr_samples, 5)
        upper = np.percentile(tr_samples, 95)
        mean = tr_samples.mean()
        # print(f"{mean=} {lower=} {upper=}")
        stats.append({
            'test_size': test_size,
            'ref_size': f'{ref_size:,}',
            'mean': mean,
            'lower': upper,
            'upper': lower
        })
    return stats

ref_sizes = [10_000, np.inf]
test_sizes = np.logspace(np.log10(1), np.log10(1_000_000), 100)

all_stats = []
for ref_size in ref_sizes:
    all_stats.extend(compute_stats(ref_size, test_sizes))

df = pd.DataFrame(all_stats)

plot = (
    ggplot(df, aes(x='test_size'))
    + geom_line(aes(y='mean', color='ref_size', fill='ref_size'), size=1)
    + geom_ribbon(aes(ymin='lower', ymax='upper', fill='ref_size', group='ref_size'), alpha=0.2)
    + scale_x_log10(breaks=[1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    + scale_y_log10()
    + labs(
        x='test sample size',
        y='expected square error', 
        color='reference Size',
        fill='reference Size'
    )
    + theme_minimal()
)
plot.save('reference-size-error.pdf')
plot.show()
