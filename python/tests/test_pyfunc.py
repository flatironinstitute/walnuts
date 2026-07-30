import pytest
import numpy as np

import walnuts as wp


## define our test target. use numba if available, for faster tests
try:
    import numba
    from numba import types
    from numba_stats import norm

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
    def logp(size, x_, grad_, lp, _):
        x = numba.carray(x_, size)
        lp[0] = norm.logpdf(x, 0.0, 1.0).sum()
        grad = numba.carray(grad_, size)
        grad[:] = -x
        return 0

except Exception:

    import scipy.stats

    def logp(x):
        return np.sum(scipy.stats.norm.logpdf(x)), -x


@pytest.mark.parametrize("MIN,MAX", [(10, 12), (77, 77), [10, 30]])
def test_warmup_requested_iter(MIN, MAX):
    fit = wp.walnuts_pyfunc(
        logp,
        num_params=2,
        min_warmup_iter=MIN,
        max_warmup_iter=MAX,
        min_sampling_iter=1,
        max_sampling_iter=1,
        save_warmup=True,
    )
    for chain in fit:
        assert MIN <= len(chain.warmup.warmup_draws) <= MAX


@pytest.mark.parametrize("MIN,MAX", [(10, 12), (77, 77), [10, 30]])
def test_sampling_requested_iter(MIN, MAX):
    fit = wp.walnuts_pyfunc(
        logp,
        num_params=2,
        min_sampling_iter=MIN,
        max_sampling_iter=MAX,
        min_warmup_iter=100,
        max_warmup_iter=100,
    )
    for chain in fit:
        assert MIN <= len(chain) <= MAX


def test_invalid_requested_iter():
    with pytest.raises(ValueError, match="min_iter must be"):
        wp.walnuts_pyfunc(
            logp, num_params=2, min_sampling_iter=100, max_sampling_iter=99
        )


def assert_draws_match(fit1, fit2):
    """Checks that the draws from two runs match up to the end of their overlapping lengths."""
    for c1, c2 in zip(fit1, fit2, strict=True):
        common_prefix = min(c1.shape[0], c2.shape[0])
        np.testing.assert_array_equal(c1[:common_prefix], c2[:common_prefix])

        assert c1.warmup.stepsize == c2.warmup.stepsize
        if c1.warmup.inv_metric is not None:
            np.testing.assert_array_equal(c1.warmup.inv_metric, c2.warmup.inv_metric)
        if c1.warmup.warmup_draws is not None:
            np.testing.assert_array_equal(
                c1.warmup.warmup_draws, c2.warmup.warmup_draws
            )


def test_seed_works():
    # turn off dynamic warmup
    warmup_length = 400
    fit1 = wp.walnuts_pyfunc(
        logp,
        num_params=4,
        seed=1234,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )
    fit2 = wp.walnuts_pyfunc(
        logp,
        num_params=4,
        seed=1234,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )

    # check that draws agree with the same seed
    assert_draws_match(fit1, fit2)

    fit3 = wp.walnuts_pyfunc(
        logp,
        num_params=4,
        seed=452,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )
    # and disagree with a different seed
    with pytest.raises(AssertionError):
        assert_draws_match(fit1, fit3)
