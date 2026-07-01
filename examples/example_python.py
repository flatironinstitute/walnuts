import os, time

import walnuts
import bridgestan


def timed(f):
    def f_timed(*args, **kwargs):
        start = time.perf_counter()
        res = f(*args, **kwargs)
        end = time.perf_counter()
        print(f"{f.__name__} took {end-start:0.6f}s")
        return res

    return f_timed


def summarize(name, fit):
    summarizer = walnuts.Summarizer(fit)
    mean = summarizer.mean()
    std = summarizer.standard_deviation()
    ess = summarizer.ess()
    r_hat = summarizer.r_hat()
    draws = summarizer._stacked.shape[0]
    print(f"{name}\tdim\tmean\tstd\tess\trhat\tdraws")
    for i in range(len(mean)):
        print(
            f"\t{i}\t{mean[i]:.4f}\t{std[i]:.4f}\t{ess[i]:.2f}\t{r_hat[i]:.4f}\t{draws}"
        )


m = bridgestan.StanModel(
    os.path.join(
        bridgestan.compile.get_bridgestan_path(), "test_models/multi/multi.stan"
    ),
    {"M": 2, "N": 0, "P": 0},
    make_args=["STAN_THREADS=1"],
)

summarize("stan", timed(walnuts.walnuts_stan)(m, seed=1234))

import scipy.stats
import numpy as np


def logp(x):
    return np.sum(scipy.stats.norm.logpdf(x)), -x


summarize("pyfunc", timed(walnuts.walnuts_pyfunc)(logp, num_params=2))

import numba
from numba_stats import norm
from numba import types


@numba.cfunc(
    types.intc(
        types.size_t,
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.voidptr,
    ),
    nopython=True,
)
def logp_numba(size, x_, grad_, lp, _):
    x = numba.carray(x_, size)
    lp[0] = norm.logpdf(x, 0.0, 1.0).sum()
    grad = numba.carray(grad_, size)
    grad[:] = -x
    return 0


summarize("numba", timed(walnuts.walnuts_pyfunc)(logp_numba, num_params=2))
