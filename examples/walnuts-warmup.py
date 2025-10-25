import pandas as pd
import plotnine as pn

# MASS MATRIX
# ======================================================================
df = pd.read_csv("walnuts-warmup-inv-mass.csv", header=None)

cols = [0, 2, 9, 29, 99, 199]
col_labels = ["1", "3", "10", "30", "100", "200"]

df_selected = df.iloc[:, cols].copy()
df_selected.columns = col_labels
df_selected["iteration"] = df_selected.index

df_long = df_selected.melt(id_vars="iteration", var_name="index", value_name="value")
df_long["index"] = pd.Categorical(df_long["index"], categories=col_labels[::-1], ordered=True)

plot = (
    pn.ggplot(df_long, pn.aes("iteration", "value", color="index"))
    + pn.geom_hline(yintercept=[1, 9, 100, 900, 10000, 40000], linetype="dashed",
                        color="red", size=0.25)
    + pn.geom_line()
    + pn.scale_x_log10()
    + pn.scale_y_log10(breaks=[1, 9, 100, 900, 10000, 40000],
                           labels=["1", "9", "100", "900", "10000", "40000"])
    + pn.labs(x="iteration", y="inverse mass", color="Index")
    + pn.theme_minimal()
)
plot.save("walnuts-warmup-inv-mass.jpg", dpi=300, width=4, height=3)


# STEP SIZE
# ======================================================================
df = pd.read_csv("walnuts-warmup-step.csv", header=None, names=["step_size"])

df["iteration"] = df.index + 1

plot = (
    pn.ggplot(df, pn.aes("iteration", "step_size"))
    + pn.geom_line()
    + pn.scale_x_log10()
    + pn.scale_y_log10()
    + pn.labs(x="iteration", y="step size")
    + pn.theme_minimal()
)
plot.save("walnuts-warmup-step.jpg", dpi=300,  width=4, height=3)
