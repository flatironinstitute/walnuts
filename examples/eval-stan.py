import json
from typing import Tuple

import numpy as np

def load_reference_moments(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        ref = json.load(fh)
    first = np.asarray(ref["first"], dtype=np.float64)
    second = np.asarray(ref["second"], dtype=np.float64)
    fourth = np.asarray(ref["fourth"], dtype=np.float64)
    return first, second, fourth

if __name__ == "__main__":
    name = 'ill-normal'
    stan_file = "models/" + name + "/" + name + ".stan"
    data_file = "models/" + name + "/" + name + "-data.json"
    moments_file = "model/" + name + "/" + name + "-moments.json"

    ref_first, ref_second, ref_fourth = load_reference_moments(moments_file)

    iter_warmup=128
    iter_sampling=10_000
    m = CmdStanModel(stan_file=stan_file)
    fit = model.sample(
        data=data_file,
        chains=1,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_engaged=True,
        show_progress=False,
        show_console=False
    )                         
    stan_vars = fit.stan_variables()
    lp = fit.method_variables()["lp__"]
    stan_vars["lp__"] = lp
    names, draws = flatten_draws(stan_vars)
    draws_squared = draws**2
    draws_fourth = draws**4
    
