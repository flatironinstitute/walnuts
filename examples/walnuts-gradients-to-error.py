import numpy as np
import pandas as pd
from pathlib import Path
from util import *

def gradients_to_error(lp_grad_calls, logp, param_draws, trial, max_error, ref_first, ref_second, ref_fourth):
    combined_draws = np.concatenate([logp[:, None], param_draws], axis=1)
    maybe_idx = draw_to_tolerance(ref_first, ref_second, ref_fourth, combined_draws, max_error)
    if maybe_idx is None:
        return -1.0;
    grads = lp_grad_calls[maybe_idx]
    return int(grads)
    
def read_draws_csv(path):
    df = pd.read_csv(path)
    lp_grad_calls = df["lp_grad_calls"].to_numpy(dtype=np.int64)
    logp = df["logp"].to_numpy(dtype=np.float64)
    param_cols = [c for c in df.columns if c not in ("lp_grad_calls", "logp")]
    params = df[param_cols].to_numpy(dtype=np.float64)
    return lp_grad_calls, logp, params

def process_walnuts_draws(dir_path, model, trials, max_error, out_file, ref_first, ref_second, ref_fourth):
    draw_dir = Path(dir_path) / model
    gradients = np.zeros(trials, dtype=int)
    for trial in range(trials):
        path = draw_dir / f"{model}-walnuts-draws-{trial}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")
        lp_grad_calls, logp, params = read_draws_csv(path)
        gradients[trial] = gradients_to_error(lp_grad_calls, logp, params, trial, max_error, ref_first, ref_second, ref_fourth)
        print(f"gradients[{trial:3d}] = {gradients[trial]:7d}")
    return gradients

if __name__ == "__main__":
    be_quiet()
    args = get_args(
        4, "walnuts-gradients-to-error.py <dir> <model> <trials> <max_error>"
    )
    dir = args[0]
    model = args[1]
    trials = int(args[2])
    max_error = float(args[3])
    print(f"WALNUTS GRAD TO ERROR {dir=};  {model=};  {trials=}")
    prefix = dir + "/" + model + "/" + model
    moments_file = prefix + "-moments.json"
    ref_first, ref_second, ref_fourth = load_reference_moments(moments_file)
    out_file = prefix + "-walnuts-gradients.csv"
    gradients = process_walnuts_draws(dir, model, trials, max_error, out_file, ref_first, ref_second, ref_fourth)
    np.savetxt(out_file, gradients, fmt="%d", header="gradients", comments="")
