import numpy as np
import pandas as pd
import plotnine as pn
import cmdstanpy as csp

def run_adaptation(stan_file, warmup_steps, output_csv):
    model = csp.CmdStanModel(stan_file=stan_file)
    D = 200
    scales = np.arange(1, D + 1)

    results = []
    for N in warmup_steps:
        fit = model.sample(
            data={"D": D, "scales": scales},
            chains=1,
            iter_warmup=N,
            iter_sampling=0,
            adapt_delta=2.0/3.0,
            save_warmup=False,
            metric="diag",
            adapt_engaged=True,
            show_progress=False,
        )

        step_size = fit.step_size[0]
        inv_mass = fit.metric
        row = np.zeros(202)
        row[0] = N
        row[1] = step_size
        row[2:202] = inv_mass
        results.append(row)
        print(f"{N = };  {step_size = };  {inv_mass = }")

    df = pd.DataFrame(
        results,
        columns=(["N", "step_size"] + [f"mass_{i+1}" for i in range(D)])
    )
    df.to_csv(output_csv, index=False)
    
warmup_steps = [100, 141, 200, 283, 400, 566, 800, 1131, 1600]
run_adaptation("../examples/diag_scale_target.stan", warmup_steps, "adapt_diag.csv")

mass_indices = ["1", "3", "10", "30", "100", "200"]
mass_columns = [f"mass_{i}" for i in mass_indices]
df = pd.read_csv("adapt_diag.csv")
df_plot = df[mass_columns].copy()
df_plot["iteration"] = df["N"]
df_long = df_plot.melt(id_vars="iteration", var_name="index", value_name="value")
df_long["index"] = pd.Categorical(df_long["index"], categories=mass_columns[::-1], ordered=True)
plot = (
    pn.ggplot(df_long, pn.aes("iteration", "value", color="index"))
    + pn.geom_line()
    + pn.scale_x_log10()
    + pn.scale_y_log10()
    + pn.labs(x="warmup iteration", y="inverse mass", color="Index")
    + pn.theme_minimal()
)
plot.save("stan-warmup-inv-mass.jpg")
    
