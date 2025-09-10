import numpy as np
import pandas as pd
import plotnine as pn

def compute_rmse(df: pd.DataFrame) -> np.ndarray:
    cumulative_means = df.expanding().mean()
    dims = np.arange(1, df.shape[1] + 1)
    standardized = cumulative_means.div(dims, axis=1)
    mse = (standardized ** 2).mean(axis=1)
    return np.sqrt(mse)

def main():
    df = pd.read_csv('walnuts-stationarity.csv', header=None)
    rmse = compute_rmse(df)
    iterations = np.arange(1, len(rmse) + 1)
    plot_df = pd.DataFrame({
        'iteration': iterations,
        'WALNUTS': rmse,
        'iid': iterations**-0.5
    })
    long_df = plot_df.melt(id_vars='iteration', var_name='source', value_name='RMSE')
    plot = (
        pn.ggplot(long_df, pn.aes(x='iteration', y='RMSE', color='source', linetype='source'))
        + pn.geom_line(size=0.5)
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        + pn.scale_color_manual(values={'iid': 'red', 'WALNUTS': 'black'})
        + pn.scale_linetype_manual(values={'iid': '--', 'WALNUTS': '-'})
        + pn.labs(
            x='Iteration',
            y='RMSE',
            title='Convergence RMSE vs. Iteration',
            color='',
            linetype=''
        )
        + pn.theme_minimal()
    )
    plot.save('walnuts-stationarity.jpg')

if __name__ == '__main__':
    main()
