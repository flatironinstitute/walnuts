import pytest
import numpy as np
import pathlib

import walnuts
import bridgestan

HERE = pathlib.Path(__file__).parent

import platform

if platform.system() == "Windows":
    import os

    bridgestan.compile.compile_model(
        HERE / "simple.stan",
        make_args=["STAN_THREADS=1"],
    )

    tbb_path = os.path.abspath(
        os.path.join(
            bridgestan.compile.get_bridgestan_path(),
            "stan",
            "lib",
            "stan_math",
            "lib",
            "tbb",
        )
    )
    os.environ["PATH"] = tbb_path + ";" + os.environ["PATH"]
    os.add_dll_directory(tbb_path)

model = bridgestan.StanModel(
    HERE / "simple.stan",
    {"N": 2},
    make_args=["STAN_THREADS=1"],
)


@pytest.mark.parametrize("MIN,MAX", [(10, 12), (77, 77), [10, 30]])
def test_warmup_requested_iter(MIN, MAX):
    fit = walnuts.walnuts_stan(
        model,
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
    fit = walnuts.walnuts_stan(
        model,
        min_sampling_iter=MIN,
        max_sampling_iter=MAX,
        min_warmup_iter=100,
        max_warmup_iter=100,
    )
    for chain in fit:
        assert MIN <= len(chain) <= MAX


def test_invalid_requested_iter():
    with pytest.raises(ValueError, match="min_iter must be"):
        walnuts.walnuts_stan(model, min_sampling_iter=100, max_sampling_iter=99)


def assert_draws_match(fit1, fit2):
    """Checks that the draws from two runs match up to the end of their overlapping lengths."""
    for c1, c2 in zip(fit1, fit2, strict=True):
        common_prefix = min(c1.data.shape[0], c2.data.shape[0])
        np.testing.assert_array_equal(c1.data[:common_prefix], c2.data[:common_prefix])

        assert c1.warmup.stepsize == c2.warmup.stepsize
        if c1.warmup.inv_metric is not None:
            np.testing.assert_array_equal(c1.warmup.inv_metric, c2.warmup.inv_metric)
        if c1.warmup.warmup_draws is not None:
            np.testing.assert_array_equal(
                c1.warmup.warmup_draws.data, c2.warmup.warmup_draws.data
            )


def test_seed_works():
    # turn off dynamic warmup
    warmup_length = 400
    fit1 = walnuts.walnuts_stan(
        model,
        seed=1234,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )
    fit2 = walnuts.walnuts_stan(
        model,
        seed=1234,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )

    # check that draws agree with the same seed
    assert_draws_match(fit1, fit2)

    fit3 = walnuts.walnuts_stan(
        model,
        seed=452,
        min_warmup_iter=warmup_length,
        max_warmup_iter=warmup_length,
        save_warmup=True,
        save_inv_metric=True,
    )
    # and disagree with a different seed
    with pytest.raises(AssertionError):
        assert_draws_match(fit1, fit3)
