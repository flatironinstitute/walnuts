import json
import numpy as np
import arviz as az
from cmdstanpy import CmdStanModel
import logging
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import cmdstanpy as csp

csp.utils.get_logger().setLevel(logging.ERROR)

import json

import json


def dump_json_sci(results: dict, path: str, sig: int = 10):
    keys = ["vars", "vars_sq", "vars_fourth", "ess", "ess_sq", "ess_fourth"]
    with open(path, "w") as f:
        f.write("{\n")
        for ki, key in enumerate(keys):
            arr = results[key]
            f.write(f'  "{key}": [\n')
            for i, v in enumerate(arr):
                comma = "," if i < len(arr) - 1 else ""
                f.write(f"    {v:.{sig}e}{comma}\n")
            end_comma = "," if ki < len(keys) - 1 else ""
            f.write(f"  ]{end_comma}\n")
        f.write("}\n")


def flatten_draws(draws_dict):
    names = []
    cols = []
    for var, arr in draws_dict.items():
        # arr.shape = (N, dim1, dim2, ...)
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
    return names, mat


def print_summary(results: dict) -> None:
    means = results["vars"]
    means_sq = results["vars_sq"]
    means_4 = results["vars_fourth"]
    ess = results["ess"]
    ess_sq = results["ess_sq"]
    ess_4 = results["ess_fourth"]

    assert (
        len(means)
        == len(means_sq)
        == len(means_4)
        == len(ess)
        == len(ess_sq)
        == len(ess_4)
    )
    D = len(means) - 1  # final variable is lp

    header = f"{'X':<11} {'E[X]':>12} {'E[X^2]':>14} {'E[X^4]':>14} {'ESS[X]':>12} {'ESS[X^2]':>12} {'ESS[X^4]':>12}"
    print(header)

    for i in range(len(means)):
        name = f"theta[{i}]" if i < D else "lp"
        print(
            f"{name:<11} {means[i]:12.5e} {means_sq[i]:14.5e} {means_4[i]:14.5e} {ess[i]:12.5e} {ess_sq[i]:12.5e} {ess_4[i]:12.5e}"
        )


def estimate(
    stan_file: str,
    data_file: str,
    out_file: str,
    min_ess_target: int,
    block_size: int,
    max_blocks: int,
    seed: int,
):
    print(f"\nSTAN PROGRAM: {stan_file = }")
    print(f"    DATA FILE: {data_file = }")
    print(f"    OUTPUT FILE: {out_file = }")
    print(f"         {min_ess_target = }")
    print(f"         {block_size = }")
    print(f"         {max_blocks = }")
    print(f"         {seed = }\n")

    model = CmdStanModel(stan_file=stan_file)
    with open(data_file) as f:
        data = json.load(f)

    sum_means = None
    sum_means_sq = None
    sum_means_4 = None
    sum_ess = None
    sum_ess_sq = None
    sum_ess_4 = None
    names = None

    print(f"{0:5d}.  min(ESS) = {0:.2e}")
    for b in range(1, max_blocks + 1):
        fit = model.sample(
            data=data,
            chains=1,
            adapt_engaged=(b == 1),
            iter_warmup=block_size if b == 1 else 0,
            iter_sampling=block_size,
            seed=seed + b,
            show_progress=False,
        )
        stan_vars = fit.stan_variables()
        lp = fit.method_variables()["lp__"]
        stan_vars["lp__"] = lp.reshape(lp.shape[0], *())
        names, mat = flatten_draws(stan_vars)
        mat_sq = mat**2
        mat_4 = mat**4

        means = mat.mean(axis=0)
        means_sq = mat_sq.mean(axis=0)
        means_4 = mat_4.mean(axis=0)

        if sum_means is None:
            K = mat.shape[1]
            sum_means = np.zeros(K)
            sum_means_sq = np.zeros(K)
            sum_means_4 = np.zeros(K)
            sum_ess = np.zeros(K)
            sum_ess_sq = np.zeros(K)
            sum_ess_4 = np.zeros(K)
        sum_means += means
        sum_means_sq += means_sq
        sum_means_4 += means_4

        posterior = {names[i]: mat[np.newaxis, :, i] for i in range(mat.shape[1])}
        posterior.update(
            {f"{names[i]}_sq": mat_sq[np.newaxis, :, i] for i in range(mat_sq.shape[1])}
        )
        posterior.update(
            {
                f"{names[i]}_fourth": mat_4[np.newaxis, :, i]
                for i in range(mat_4.shape[1])
            }
        )
        idata = az.from_dict(posterior=posterior)
        ess = az.ess(idata)

        for i, name in enumerate(names):
            sum_ess[i] += float(ess[name].values)
            sum_ess_sq[i] += float(ess[f"{name}_sq"].values)
            sum_ess_4[i] += float(ess[f"{name}_fourth"].values)

        min_ess = min(np.min(sum_ess), np.min(sum_ess_sq), np.min(sum_ess_4))
        if b % 10 == 0:
            print(f"{b:5d}.  min(ESS) = {min_ess:.2e}")
        if min_ess > min_ess_target:
            print(f"{b:5d}.  min(ESS) = {min_ess:.2e}")
            print("\n***** ACHIEVED MINIMUM ESS TARGET *****\n")
            break

    num_blocks = b
    avg_means = sum_means / num_blocks
    avg_means_sq = sum_means_sq / num_blocks
    avg_means_4 = sum_means_4 / num_blocks

    ordered = [n for n in names if n != "lp__"] + ["lp__"]
    fmt = lambda x: float(f"{x:.8g}")
    vars_ = [fmt(avg_means[names.index(n)]) for n in ordered]
    vars_sq = [fmt(avg_means_sq[names.index(n)]) for n in ordered]
    vars_fourth = [fmt(avg_means_4[names.index(n)]) for n in ordered]

    def tolist(arr):
        return [float(arr[names.index(n)]) for n in ordered]

    out_dict = {
        "vars": tolist(vars_),
        "vars_sq": tolist(vars_sq),
        "vars_fourth": tolist(vars_fourth),
        "ess": tolist(sum_ess),
        "ess_sq": tolist(sum_ess_sq),
        "ess_fourth": tolist(sum_ess_4),
    }

    print_summary(out_dict)
    dump_json_sci(out_dict, out_file)


if __name__ == "__main__":
    stan_file = "../examples/diag_scale_target.stan"
    data_file = "../examples/diag_scale_target.json"
    out_file = "../examples/diag_scale_target_out.json"
    min_ess_target = 1e5
    block_size = 10_000
    max_blocks = 10_000
    seed = 643889
    estimate(
        stan_file, data_file, out_file, min_ess_target, block_size, max_blocks, seed
    )
