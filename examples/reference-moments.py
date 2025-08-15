import numpy as np
import cmdstanpy as csp
from util import *

def adapt(
    model: csp.CmdStanModel, data: dict, iter_warmup: int, seed: int
) -> tuple[dict, float, np.ndarray]:
    fit = model.sample(
        data=data,
        seed=seed,
        adapt_engaged=True,
        chains=1,
        iter_warmup=iter_warmup,
        iter_sampling=10,
        show_progress=False,
        show_console=False,
    )
    vars_dict = fit.stan_variables()
    last_draw = {k: v[-1] for k, v in vars_dict.items()}
    return fit.step_size[0], fit.metric[0], last_draw

def estimate(
    stan_file: str,
    data_file: str,
    out_file: str,
    min_ess_target: int,
    block_size: int,
    max_blocks: int,
    initial_warmup: int,
    per_block_burnin: int,
    seed: int,
):
    print(f"\nSTAN PROGRAM: {stan_file = }")
    print(f"    DATA FILE: {data_file = }")
    print(f"    OUTPUT FILE: {out_file = }")
    print(f"         {min_ess_target = }")
    print(f"         {block_size = }")
    print(f"         {max_blocks = }")
    print(f"         {seed = }\n")

    print("COMPILING MODEL")
    model = csp.CmdStanModel(stan_file=stan_file)
    print("LOADING DATA")
    data = get_model_data(data_file)

    print("ADAPTING")
    step_size, metric, state = adapt(model, data, initial_warmup, seed)
    norm2_metric = np.linalg.norm(metric)
    print(f"    {step_size = };  ||inv(mass)|| = {norm2_metric}")

    K = metric.size + 1
    sum_means = np.zeros(K)
    sum_means_sq = np.zeros(K)
    sum_means_4 = np.zeros(K)
    sum_ess_first = np.zeros(K)
    sum_ess_second = np.zeros(K)
    sum_ess_fourth = np.zeros(K)

    print("SAMPLING")
    print(f"{0:5d}.  min(ESS) = {0:.1e}")
    for b in range(1, max_blocks + 1):
        fit = model.sample(
            data=data,
            chains=1,
            adapt_engaged=0,
            iter_warmup=per_block_burnin,
            iter_sampling=block_size,
            seed=seed + b,
            step_size=step_size,
            metric=[{"inv_metric": metric}],
            inits=state,
            show_progress=False,
            show_console=False,
        )
        mat = lp_params(fit)
        mat_sq = mat ** 2
        mat_4 = mat ** 4

        means = mat.mean(axis=0)
        means_sq = mat_sq.mean(axis=0)
        means_4 = mat_4.mean(axis=0)

        sum_means += means
        sum_means_sq += means_sq
        sum_means_4 += means_4

        sum_ess_first += ess_per_col(mat)
        sum_ess_second += ess_per_col(mat_sq)
        sum_ess_fourth += ess_per_col(mat_4)

        min_ess = min(
            np.min(sum_ess_first), np.min(sum_ess_second), np.min(sum_ess_fourth)
        )
        print(f"{b:5d}.  min(ESS) = {min_ess:.1e}")
        if min_ess > min_ess_target:
            print("\n***** ACHIEVED MINIMUM ESS TARGET *****\n")
            break

    avg_means = sum_means / b
    avg_means_sq = sum_means_sq / b
    avg_means_4 = sum_means_4 / b
    out_dict = {
        "first": avg_means,
        "second": avg_means_sq,
        "fourth": avg_means_4,
        "ess_first": sum_ess_first,
        "ess_second": sum_ess_second,
        "ess_fourth": sum_ess_fourth,
    }
    dump_json_sci(out_dict, out_file)


if __name__ == "__main__":
    args = get_args(1, "reference-moments.py model_name")
    name = args[0]
    stan_file = "models/" + name + "/" + name + ".stan"
    data_file = "models/" + name + "/" + name + "-data.json"
    moments_file = "models/" + name + "/" + name + "-moments.json"
    min_ess_target = 1e5
    block_size = 10_000
    max_blocks = 1_000  # block_size * max_blocks -> 10M iters max
    initial_warmup = 50_000
    per_block_burnin = 100  # init off, so only burnin
    seed = 643889
    stop_griping()
    estimate(
        stan_file,
        data_file,
        moments_file,
        min_ess_target,
        block_size,
        max_blocks,
        initial_warmup,
        per_block_burnin,
        seed,
    )
