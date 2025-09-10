import json
import numpy as np
from scipy.stats import norm

def sample_funnel(rng, D):
    two_log_sigma = rng.normal(loc=0.0, scale=3.0)
    sigma = np.exp(two_log_sigma / 2.0)
    alphas = rng.normal(loc=0.0, scale=sigma, size=(D - 1))
    logp = (norm.logpdf(two_log_sigma, loc=0.0, scale=3.0)
                + norm.logpdf(alphas, loc=0.0, scale=sigma).sum())
    return logp, two_log_sigma, alphas

def estimate_moments(rng, D, M):
    sum1, sum2, sum4 = np.zeros(D + 1, dtype=float), np.zeros(D + 1,
    dtype=float), np.zeros(D + 1, dtype=float)
    for _ in range(M):
        logp, log_sigma, alphas = sample_funnel(rng, D)
        x = np.array([logp, log_sigma, *alphas])
        sum1 += x
        sum2 += x**2
        sum4 += x**4
    m1, m2, m4 = sum1 / M, sum2 / M, sum4 / M	
    return m1, m2, m4

if __name__ == "__main__":
    seed = 1234
    D = 10
    M = 10_000
    rng = np.random.default_rng(seed)
    m1, m2, m4 = estimate_moments(rng, D, M)
    out = {
        "first": m1.tolist(),
        "second": m2.tolist(),
        "fourth": m4.tolist()
    }
    print(out)
    with open("funnel-data.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
