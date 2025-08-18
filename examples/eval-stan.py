import json

import numpy as np
from util import *

import numpy as np


def draw_to_tolerance(
    ref_first: np.ndarray,
    ref_second: np.ndarray,
    ref_fourth: np.ndarray,
    draws_first: np.ndarray,
    draws_second: np.ndarray,
    draws_fourth: np.ndarray,
    tol: float = 0.1,
):
    n_rows = draws_first.shape[0]
    denom = np.arange(1, n_rows + 1, dtype=float).reshape(-1, 1)

    run_first = np.cumsum(draws_first, axis=0) / denom
    run_second = np.cumsum(draws_second, axis=0) / denom

    err_first = np.abs(run_first - ref_first.reshape(1, -1))
    err_second = np.abs(run_second - ref_second.reshape(1, -1))

    sd_first =  np.sqrt(ref_second - ref_first**2)
    std_err_first = err_first / sd_first.reshape(1, -1)
    sd_second = np.sqrt(ref_fourth - ref_second**2)
    std_err_second = err_second / sd_second.reshape(1, -1)

    ok_rows = (
        (std_err_first < tol).all(axis=1)
        & (std_err_second < tol).all(axis=1)
    )

    idx = np.flatnonzero(ok_rows)
    return int(idx[0]) if idx.size else None


def num_leapfrogs(fit, iter_index):
    n_leapfrog = fit.method_variables()["n_leapfrog__"]
    if iter_index < 0 or iter_index >= n_leapfrog.shape[0]:
        raise IndexError("iter_index out of range.")
    return int(n_leapfrog[: iter_index + 1].sum())


if __name__ == "__main__":
    stop_griping()
    args = get_args(2, "eval-stan.py model_name iter_warmup")
    print(f"{args=}")
    name = args[0]
    iter_warmup = int(args[1])
    stan_file = "models/" + name + "/" + name + ".stan"
    data_file = "models/" + name + "/" + name + "-data.json"
    moments_file = "models/" + name + "/" + name + "-moments.json"
    print(f"model {name=};  {iter_warmup=}")

    ref_first, ref_second, ref_fourth = load_reference_moments(moments_file)

    seed = 8474364
    iter_sampling = 5000
    model = csp.CmdStanModel(stan_file=stan_file)
    b_head = "trial"
    it_head = "iteration"
    leap_head = "leapfrog"
    print(f"{b_head:5s}, {it_head:9s}, {leap_head:8s}")
    for b in range(128):
        fit = model.sample(
            data=data_file,
            chains=1,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed + b,
            sig_figs=10,
            save_warmup=True,
            adapt_engaged=True,
            show_progress=False,
            show_console=False
            )
        draws_first = lp_params(fit)
        draws_second = draws_first**2
        draws_fourth = draws_first**4
        maybe_idx = draw_to_tolerance(
            ref_first, ref_second, ref_fourth, draws_first, draws_second, draws_fourth
            )
        if maybe_idx is not None:
            steps = num_leapfrogs(fit, maybe_idx)
            print(f"{b:5d}, {maybe_idx:9d}, {steps:8d}")
        else:
            print("{b:5d}, {-1:9d}, {-1:8d}")
