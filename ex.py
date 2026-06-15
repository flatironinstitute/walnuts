from python import walnuts_pyfunc, walnuts_stan

import os
import bridgestan

m = bridgestan.StanModel(
    os.path.join(
        bridgestan.compile.get_bridgestan_path(), "test_models/multi/multi.stan"
    ),
    {"M": 2, "N": 0, "P": 0},
    make_args=["STAN_THREADS=1"],
)

import scipy.stats
import numpy as np


def logp(x):
    return np.sum(scipy.stats.norm.logpdf(x)), -x


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


# print([(c['alpha'].mean(axis=0), c['alpha'].shape) for c in walnuts_stan(m, seed=1234, num_chains=1)])
# print([(c.mean(axis=0), c.shape) for c in walnuts_pyfunc(logp, num_params=2)])
# print([(c.mean(axis=0), c.shape) for c in walnuts_pyfunc(logp_numba, num_params=2)])
