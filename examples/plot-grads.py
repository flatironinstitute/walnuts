import pandas as pd
import plotnine as pn
from util import *

def rnd(x: int) -> str:
    return int(x - (x % 1000))

def plot_grads_facet(model):
    nuts = pd.read_csv(f"models/{model}/{model}-nuts-gradients.csv")
    nuts["method"] = "NUTS"
    walnuts = pd.read_csv(f"models/{model}/{model}-walnuts-gradients.csv")
    walnuts["method"] = "WALNUTS"
    df = pd.concat([nuts, walnuts], ignore_index=True)
    x_min, x_max = df["gradients"].min(), df["gradients"].max()
    means = df.groupby("method", as_index=False)["gradients"].mean().rename(columns={"gradients": "mean"})
    plot = (
        pn.ggplot(df, pn.aes(x="gradients"))
        + pn.geom_histogram(pn.aes(y="..density.."), fill="white", bins=32, color="black")
        # + pn.geom_density(adjust=2.0, trim=True, color="red", size=1.2)
        + pn.geom_vline(pn.aes(xintercept="mean"), data=means, color="blue", size=1.2)
        + pn.facet_grid("method ~ .")
        + pn.scale_x_continuous(limits=(x_min, x_max),
                                    breaks=[rnd(m) for m in means['mean']])
        + pn.scale_y_continuous(breaks=[])
        + pn.labs(x=f"gradient calls to error tolerance", y="", title=f"Model: {model}")
        # + pn.theme_minimal()
        + pn.theme(axis_text_y=pn.element_blank(), axis_ticks_major_y=pn.element_blank())
        # + pn.theme(panel_spacing_y=0.05)
        + pn.theme(axis_title_x=pn.element_text(margin={'t': 20}))
        + pn.theme(strip_text=pn.element_text(size=12))
    )
    out = f"models/{model}/{model}-nuts-vs-walnuts-grads.jpg"
    plot.save(out, dpi=200)
    plot.show()

if __name__ == "__main__":
    be_quiet()
    args = get_args(1, "plot-grads-facet.py model")
    plot_grads_facet(args[0])
