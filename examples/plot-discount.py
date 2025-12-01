import numpy as np
import pandas as pd
import plotnine as pn

mass_iter_offset = 4
n = np.arange(1, 1000)
alpha = 1 - 1 / (mass_iter_offset + n)

df = pd.DataFrame({'n': n, 'alpha': alpha})

plot = (
    pn.ggplot(df, pn.aes(x='n', y='alpha'))
    + pn.geom_line()
    + pn.scale_x_log10()
    + pn.scale_y_continuous(breaks=[0.5, 0.6, 0.7, 0.8, 0.9,
    0.95, 0.99, 1.0])
    + pn.labs(x='n', y=r'$\alpha^{(n)}$')
    + pn.theme_minimal()
    + pn.theme(
        axis_title=pn.element_text(size=16),
        axis_text=pn.element_text(size=12)
    )
)

plot.save('discount-schedule.jpg', dpi=300)
print(plot)
