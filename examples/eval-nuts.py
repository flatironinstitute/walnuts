import json
import numpy as np
from util import *

def num_leapfrogs(fit, iter_index, iter_warmup):
    n_leapfrog = fit.method_variables()["n_leapfrog__"]
    last_index = iter_index + iter_warmup + 1
    if last_index < 0 or last_index >= n_leapfrog.shape[0]:
        raise IndexError("iter_index out of range.")
    return int(n_leapfrog[ :last_index].sum())


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
    print("")
    print(f"EVALUATING NUTS")
    print("")
    print(f"model={name};  {max_error=};  {num_trials=};  {iter_warmup=};  {iter_sampling=}")
    stan_file = prefix + ".stan"
    data_file = prefix + "-data.json"
    moments_file = prefix + "-moments.json"
    gradients_file = prefix + "-nuts-gradients.csv"
    print(f"Model Stan file: {stan_file}")
    print(f"Data JSON file: {data_file}")
    print(f"Moments JSON file: {moments_file}")
    print(f"Output CSV file: {gradients_file}")

    print(f"\nLoading Reference Moments")
    ref_first, ref_second, ref_fourth = load_reference_moments(moments_file)

    seed = 8474364

    print(f"\nCompling Model")
    model = csp.CmdStanModel(stan_file=stan_file)
    b_head = "trial"
    it_head = "iteration"
    leap_head = "leapfrog"
    print("")
    print(f"{b_head:5s}, {it_head:9s}, {leap_head:8s}")
    gradients = num_trials * [-1]
    for trial in range(num_trials):
        fit = model.sample(
            data=data_file,
            chains=1,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed + 17 * trial,
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
        gradients[trial] = num_leapfrogs(fit, maybe_idx, iter_warmup)
        print(f"{trial:5d}, {maybe_idx:9d}, {gradients[trial]:8d}")
    with open(gradients_file, "w") as f:
        f.write("gradients\n")
        f.writelines(f"{g}\n" for g in gradients)
    print("")
    print(f"gradients = {gradients}")
    print("")
    print(f"Finished.")
