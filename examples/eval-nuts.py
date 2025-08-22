import json

import numpy as np
from util import *

import numpy as np


def draw_to_tolerance(
    ref_first: np.ndarray,
    ref_second: np.ndarray,
    ref_fourth: np.ndarray,
    draws_first: np.ndarray,
    max_error: float,
):
    n_rows = draws_first.shape[0]
    denom = np.arange(1, n_rows + 1, dtype=float).reshape(-1, 1)

    draws_second = draws_first**2
    draws_fourth = draws_first**4

    run_first = np.cumsum(draws_first, axis=0) / denom
    run_second = np.cumsum(draws_second, axis=0) / denom

    err_first = np.abs(run_first - ref_first.reshape(1, -1))
    err_second = np.abs(run_second - ref_second.reshape(1, -1))

    sd_first = np.sqrt(ref_second - ref_first**2)
    std_err_first = err_first / sd_first.reshape(1, -1)
    sd_second = np.sqrt(ref_fourth - ref_second**2)
    std_err_second = err_second / sd_second.reshape(1, -1)

    ok_rows = (std_err_first < max_error).all(axis=1) & (
        std_err_second < max_error
    ).all(axis=1)

    idx = np.flatnonzero(ok_rows)
    return int(idx[0]) if idx.size else None


def num_leapfrogs(fit, iter_index):
    n_leapfrog = fit.method_variables()["n_leapfrog__"]
    if iter_index < 0 or iter_index >= n_leapfrog.shape[0]:
        raise IndexError("iter_index out of range.")
    return int(n_leapfrog[: iter_index + 1].sum())


if __name__ == "__main__":
    be_quiet()
    args = get_args(
        5, "eval-nuts.py <model> <max_err> <num_trials> <iter_warmup> <iter_sampling>"
    )
    name = args[0]
    max_error = float(args[1])
    num_trials = int(args[2])
    iter_warmup = int(args[3])
    iter_sampling = int(args[4])
    prefix = "models/" + name + "/" + name
    print(f"MODEL {name=};  {max_error=};  {num_trials=};  {iter_warmup=};  {iter_sampling=}")
    print(f"Writing to path: {prefix}")
    stan_file = prefix + ".stan"
    data_file = prefix + "-data.json"
    moments_file = prefix + "-moments.json"
    gradients_file = prefix + "-nuts-gradients.csv"

    ref_first, ref_second, ref_fourth = load_reference_moments(moments_file)

    seed = 8474364
    model = csp.CmdStanModel(stan_file=stan_file)
    b_head = "trial"
    it_head = "iteration"
    leap_head = "leapfrog"
    print(f"{b_head:5s}, {it_head:9s}, {leap_head:8s}")
    gradients = num_trials * [-1]
    for trial in range(num_trials):
        fit = model.sample(
            data=data_file,
            chains=1,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed + trial,
            sig_figs=10,
            save_warmup=True,
            adapt_engaged=True,
            show_progress=False,
            show_console=False,
        )
        draws_first = lp_params(fit)
        maybe_idx = draw_to_tolerance(
            ref_first, ref_second, ref_fourth, draws_first, max_error
        )
        if maybe_idx is None:
            continue
        gradients[trial] = num_leapfrogs(fit, maybe_idx)
        print(f"{trial:5d}, {maybe_idx:9d}, {gradients[trial]:8d}")
    with open(gradients_file, "w") as f:
        f.write("gradients\n")
        f.writelines(f"{g}\n" for g in gradients)        
    print(f"FINISHED; gradients = {gradients}")
