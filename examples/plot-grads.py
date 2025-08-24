#!/usr/bin/env python3
import pandas as pd
import plotnine as pn
from util import *

def plot_grads(csv_path, model):
    df = pd.read_csv(csv_path)
    plot = (
        pn.ggplot(df, pn.aes(x="gradients"))
        + pn.geom_histogram(pn.aes(y="..density.."), fill="lightyellow", color="black", alpha=0.2)
        + pn.geom_density(adjust=2.0, trim=True, color="red", size=1.2)
        + pn.geom_vline(xintercept=df["gradients"].mean(), color="blue", size=1.2)
        + pn.labs(x="gradient calls", y="", title=f"gradients to error ({model} model)")
        + pn.scale_y_continuous(breaks=[])
        + pn.theme(axis_text_y = pn.element_blank(), axis_ticks_major_y = pn.element_blank())
        + pn.theme_minimal()
    )
    plot.save(f'models/{model}/{model}-nuts-grads.jpg')
    plot.show()

if __name__ == "__main__":
    be_quiet()
    args = get_args(1, "plot-grads.py model")
    model = args[0]
    csv_path = f"models/{model}/{model}-nuts-gradients.csv"
    plot_grads(csv_path, model)
