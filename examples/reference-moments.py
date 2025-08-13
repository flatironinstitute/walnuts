import json
import logging
import warnings

import numpy as np
import xarray as xr

import arviz as az
import cmdstanpy as csp

# restrict output to errors
warnings.simplefilter(action="ignore", category=FutureWarning)
csp.utils.get_logger().setLevel(logging.ERROR)


def dump_json_sci(results: dict, path: str, sig: int = 5):
    with open(path, "w") as f:
        f.write("{\n")
        for ki, (key, arr) in enumerate(results.items()):
            f.write(f'  "{key}": [\n')
            for i, v in enumerate(arr):
                comma = "," if i < len(arr) - 1 else ""
                f.write(f"    {v:.{sig}e}{comma}\n")
            end_comma = "," if ki < len(results) - 1 else ""
            f.write(f"  ]{end_comma}\n")
        f.write("}\n")
        

def flatten_draws(draws_dict):
    names = []
    cols = []
    for var, arr in draws_dict.items():
        # arr.shape = (N, dim1, dim2, ...)
        print(f"{var=};  {arr.shape=}")
        N = arr.shape[0]
        rest = arr.shape[1:]
        flat = arr.reshape(N, -1)  # shape (N, prod(rest))
        for j in range(flat.shape[1]):
            if rest:
                idx = np.unravel_index(j, rest)
                idx_str = ",".join(str(i) for i in idx)
                name = f"{var}[{idx_str}]"
            else:
                name = var
            names.append(name)
        cols.append(flat)
    mat = np.hstack(cols)  # (N, K)
    print(f"     {mat.shape=}")
    return names, mat


def adapt(model: csp.CmdStanModel,
              data: dict,
              iter_warmup: int,
              seed: int) -> tuple[dict, float, np.ndarray]:
    fit = model.sample(
        data=data,
        seed=seed,
        adapt_engaged=True,
        chains=1,
        iter_warmup=iter_warmup,
        iter_sampling=10,
        show_progress=False,
        show_console=False
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
    per_block_warmup: int,
    seed: int,
):
    print(f"\nSTAN PROGRAM: {stan_file = }")
    print(f"    DATA FILE: {data_file = }")
    print(f"    OUTPUT FILE: {out_file = }")
    print(f"         {min_ess_target = }")
    print(f"         {block_size = }")
    print(f"         {max_blocks = }")
    print(f"         {seed = }\n")

    model = csp.CmdStanModel(stan_file=stan_file)
    with open(data_file) as f:
        data = json.load(f)

    print("ADAPTING")
    step_size, metric, state = adapt(model, data, initial_warmup, seed)
    norm2_metric = np.linalg.norm(metric)
    print(f"    {step_size = };  ||inv(metric)|| = {norm2_metric}")

    K = metric.size + 1
    sum_means = np.zeros(K)
    sum_means_sq = np.zeros(K)
    sum_means_4 = np.zeros(K)
    sum_ess_first = np.zeros(K)
    sum_ess_second = np.zeros(K)
    sum_ess_fourth = np.zeros(K)
    names = None

    print("SAMPLING")
    print(f"{0:5d}.  min(ESS) = {0:.2e}")
    for b in range(1, max_blocks + 1):
        fit = model.sample(
            data=data,
            chains=1,
            adapt_engaged=0,
            iter_warmup=per_block_warmup,
            iter_sampling=block_size,
            seed=seed + b,
            step_size=step_size,
            metric= [{'inv_metric': metric}],
            inits=state,
            show_progress=False,
            show_console=False,
        )
        stan_vars = fit.stan_variables()
        lp = fit.method_variables()["lp__"]
        stan_vars["lp__"] = lp
        names, mat = flatten_draws(stan_vars)
        mat_sq = mat**2
        mat_4 = mat**4

        means = mat.mean(axis=0)
        means_sq = mat_sq.mean(axis=0)
        means_4 = mat_4.mean(axis=0)
        sum_means += means
        sum_means_sq += means_sq
        sum_means_4 += means_4

        def ess_per_col(a: np.ndarray) -> np.ndarray:
            da = xr.DataArray(a[np.newaxis, :, :], dims=("chain", "draw", "var"))
            ds = az.ess(da, method="bulk")
            data_var = next(iter(ds.data_vars))
            vec = ds[data_var].values
            return np.asarray(np.squeeze(vec), dtype=np.float64)        

        sum_ess_first  += ess_per_col(mat)
        sum_ess_second += ess_per_col(mat_sq)
        sum_ess_fourth += ess_per_col(mat_4)            

        min_ess = min(np.min(sum_ess_first), np.min(sum_ess_second), np.min(sum_ess_fourth))
        print(f"{b:5d}.  min(ESS) = {min_ess:.2e}")
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
        "ess_fourth": sum_ess_fourth
    }
    dump_json_sci(out_dict, out_file)


if __name__ == "__main__":
    model = 'ill-normal'
    stan_file = "models/" + model + "/" + model + ".stan"
    data_file = "models/" + model + "/" + model + "-data.json"
    moments_file = "models/" + model + "/" + model + "-moments.json"
    min_ess_target = 1e4
    block_size = 10_000
    max_blocks = 1_000
    initial_warmup=50_000
    per_block_warmup=5_000
    seed = 643889
    estimate(stan_file, data_file, moments_file, min_ess_target,
                 block_size, max_blocks, initial_warmup, per_block_warmup, seed
    )
