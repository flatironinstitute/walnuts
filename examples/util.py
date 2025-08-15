import json
import logging
import warnings

import numpy as np
import xarray as xr

import arviz as az
import cmdstanpy as csp

def stop_griping():
    """Restrict warnings and logging to errors."""
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

        
def lp_params(fit):
    # retains column 0 (lp__) and columns {7, 8, ... } (stan variables)
    # deletes NUTS diagnostics columns {1, ..., 6}
    draws = fit.draws(inc_warmup=False, concat_chains=True)
    return draws[:, np.r_[0, 7 : draws.shape[1]]]


def ess_per_col(a: np.ndarray) -> np.ndarray:
    da = xr.DataArray(a[np.newaxis, :, :], dims=("chain", "draw", "var"))
    ds = az.ess(da, method="bulk")
    data_var = next(iter(ds.data_vars))
    vec = ds[data_var].values
    return np.asarray(np.squeeze(vec), dtype=np.float64)        
