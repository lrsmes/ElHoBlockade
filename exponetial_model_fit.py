import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as statistics


def goodness_of_fit(y_data, y_fit):
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)  # Residual sum of squares (RSS)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared
    return ss_res, r_squared


def f_test(y_data, y_simple_fit, y_complex_fit, p_simple, p_complex):
    # Number of observations
    n = len(y_data)

    # Residual sum of squares for simpler model
    rss_simple = np.sum((y_data - y_simple_fit) ** 2)

    # Residual sum of squares for more complex model
    rss_complex = np.sum((y_data - y_complex_fit) ** 2)

    # F-statistic formula
    numerator = (rss_simple - rss_complex) / (p_complex - p_simple)
    denominator = rss_complex / (n - p_complex)
    F_stat = numerator / denominator

    # Degrees of freedom
    df_numerator = p_complex - p_simple
    df_denominator = n - p_complex

    # Calculate the p-value using the F-distribution
    p_value = 1 - statistics.f.cdf(F_stat, df_numerator, df_denominator)

    return F_stat, p_value


def plot_data(ax, x, y, y_err=None ,marker='o', color='indianred', xlabel='x', ylabel='y'):
    ax.errorbar(x, y, y_err, fmt=marker, markersize=6 * (np.pi / 4)**-1, markerfacecolor=color,
                 ecolor='black', capsize=2, markeredgecolor='black')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.5)


def model(x, A, B, lamda):
    return A * np.exp(-1 * x / lamda) + B


def plot_parameter_distributions(trace, param_names=None, title="Parameter Distributions"):
    """
    Plots the distribution of each parameter in a model from prior or posterior samples.

    Parameters:
    - trace: InferenceData (e.g., output from pm.sample_prior_predictive() or pm.sample()).
    - param_names: List of parameter names to plot. If None, plots all available parameters.
    - title: Title of the plot.

    Usage:
    - To plot priors: plot_parameter_distributions(prior_pred.prior)
    - To plot posteriors: plot_parameter_distributions(trace.posterior)
    """
    if param_names is None:
        param_names = list(trace.keys())  # Automatically extract available parameter names

    num_params = len(param_names)
    num_cols = 3  # Number of columns in the plot grid
    num_rows = (num_params + num_cols - 1) // num_cols  # Compute needed rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, param in enumerate(param_names):
        if param in trace:
            az.plot_dist(trace[param].values.flatten(), ax=axes[i], kind="hist", label=param)
            axes[i].set_title(param)
            axes[i].legend()
        else:
            axes[i].axis("off")  # Hide empty subplots

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def load_data(path_1, path_2, path_3):
    x = np.load(path_1)
    y = np.load(path_2)
    y_err = np.load(path_3)
    return [x, y, y_err]

def load_multiple_data(path_1, path_2, path_3):
    x = np.load(path_1)
    y = np.transpose(np.load(path_2))
    y_err = np.transpose(np.load(path_3))
    print(np.shape(y))
    print(np.shape(x))
    return [x, y, y_err]

def fitting_model(x1, x2, y1, y2,
                  paramsA=(1.0, 0.3, 1.0, 0.3),
                  paramsB=(0.5, 0.3, 0.5, 0.3),
                  paramslamda1=(1.5, 0.04),
                  paramsr=(16.0, 1.5),
                  paramssigma=(0.2, 0.2)):

    with pm.Model() as model:
        A1 = pm.Normal("A1", sigma=paramsA[0], mu=paramsA[1])
        A2 = pm.Normal("A2", sigma=paramsA[2], mu=paramsA[3])

        B1 = pm.Normal("B1", sigma=paramsB[0], mu=paramsB[1])
        B2 = pm.Normal("B2", sigma=paramsB[2], mu=paramsB[3])

        lamda1 = pm.Weibull("lamda1", alpha=paramslamda1[0], beta=paramslamda1[1])
        r = pm.Normal('r', mu=paramsr[0], sigma=paramsr[1])
        lamda2 = pm.Deterministic('lamda2', r * lamda1)

        y1_pred = pm.Deterministic("y1_pred", A1 * pm.math.exp(-1 * x1 / lamda1) + B1)
        y2_pred = pm.Deterministic("y2_pred", A2 * pm.math.exp(-1 * x2 / lamda2) + B2)

        sigma1 = pm.HalfNormal("sigma1", sigma=paramssigma[0])
        sigma2 = pm.HalfNormal("sigma2", sigma=paramssigma[1])
        y1_obs = pm.Normal("y1_obs", mu=y1_pred, sigma=sigma1, observed=y1)
        y2_obs = pm.Normal("y2_obs", mu=y2_pred, sigma=sigma2, observed=y2)

        prior_pred = pm.sample_prior_predictive(samples=10000)

        trace = pm.sample(4000, tune=4000, return_inferencedata=True, target_accept=0.99, chains=4)
    return prior_pred, trace

def model_analyz(data_1, data_2, prior_pred, trace, plot_density=False):
    # Extract mean values of each parameter
    posterior = az.summary(trace, kind="stats")
    posterior_means = az.summary(trace, kind="stats")["mean"]
    posterior_std = posterior["sd"]
    print("\n Posterior Estimates:")
    print(posterior)



    x1_model = np.arange(0.0, 1.1 * np.max(data_1[0]), 0.001)
    y_1_model = model(x1_model, posterior_means["A1"], posterior_means["B1"], posterior_means["lamda1"])

    x2_model = np.arange(0.0, 1.1 * np.max(data_2[0]), 0.001)
    y_2_model = model(x2_model, posterior_means["A2"], posterior_means["B2"], posterior_means["lamda2"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    ax1, ax2, ax3, ax4 = axes.flatten()
    plot_data(ax1, data_1[0], data_1[1], data_1[2], xlabel=r't$_{\mathrm{read}}$ ($\mu$s)', ylabel='$S_{read}$/$S_{ini}$')
    ax1.plot(x1_model, y_1_model, ls=':', color='grey', lw=2)
    plot_data(ax2, data_2[0], data_2[1], data_2[2], xlabel=r't$_{\mathrm{read}}$ ($\mu$s)', ylabel='$S_{read}$/$S_{ini}$')
    ax2.plot(x2_model, y_2_model, ls=':', color='grey', lw=2)
    ax2.text(0.98, 0.98,
             rf'$\tau_1 / \tau_2$: {posterior_means["r"]:.1f} $\pm$ {posterior_std["r"]:.1f}',
             transform=ax2.transAxes,
             fontsize=7.5, verticalalignment="top", horizontalalignment="right",
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))


    r_squared_fit = goodness_of_fit(data_1[1], model(data_1[0], posterior_means["A1"], posterior_means["B1"],
                                                     posterior_means["lamda1"]))[1]
    plot_data(ax3, data_1[0],
              model(data_1[0], posterior_means["A1"], posterior_means["B1"], posterior_means["lamda1"]) - data_1[1],
              color='grey', xlabel=r't$_{\mathrm{read}}$ ($\mu$s)', ylabel='residuals')
    ax3.axhline(0, ls='-.', color='black')
    ax3.text(0.98, 0.98, '$R^{2}$: ' + f'{r_squared_fit:.3f}', transform=ax3.transAxes,
            fontsize=7.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    r_squared_fit = goodness_of_fit(data_2[1], model(data_2[0], posterior_means["A2"], posterior_means["B2"],
                                                     posterior_means["lamda2"]))[1]
    plot_data(ax4, data_2[0],
              model(data_2[0], posterior_means["A2"], posterior_means["B2"], posterior_means["lamda2"]) - data_2[1],
              color='grey', xlabel=r't$_{\mathrm{read}}$ ($\mu$s)', ylabel='residuals')
    ax4.axhline(0, ls='-.', color='black')
    ax4.text(0.98, 0.98, '$R^{2}$: ' + f'{r_squared_fit:.3f}', transform=ax4.transAxes,
            fontsize=7.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.tight_layout()
    plt.show()
    if plot_density:
        # Plot prior distributions
        plot_parameter_distributions(prior_pred.prior, title="Prior Distributions")

        # Plot posterior distributions
        plot_parameter_distributions(trace.posterior, title="Posterior Distributions")


def main(p0, path_dir='/Users/hubert.D/Documents/0T_ratios/', single=True):

    if single:
        dp1 = np.array([path_dir + 'tread_transport.npy', path_dir + 'ratios_transport.npy', path_dir + 'ratios_err_transport.npy'])
        dp2 = np.array(
            [path_dir + 'tread_blockade.npy', path_dir + 'ratios_blockade.npy', path_dir + 'ratios_err_blockade.npy'])
        data_1 = load_data(*dp1)
        data_2 = load_data(*dp2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        plot_data(ax1, data_1[0], data_1[1], data_1[2])
        plot_data(ax2, data_2[0], data_2[1], data_2[2])
        plt.show()

        prior_pred, trace = fitting_model(data_1[0], data_2[0], data_1[1], data_2[1], *p0)

        y1_prior_samples = prior_pred.prior["y1_pred"].values[0]  # Shape: (1000, len(x1))
        y2_prior_samples = prior_pred.prior["y2_pred"].values[0]  # Shape: (1000, len(x1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        print("Shape of x1:", data_1[0].shape)
        print("Shape of y1_prior_samples:", y1_prior_samples.shape)
        print("Shape of y1_prior_samples[i, :]:", y1_prior_samples[0, :].shape)

        for i in range(5):
            plot_data(ax1, data_1[0], y1_prior_samples[i, :])
            plot_data(ax2, data_2[0], y2_prior_samples[i, :])

        plt.show()

        model_analyz(data_1, data_2, prior_pred, trace)

    else:
        dp1 = np.array([path_dir + 'tread_transport.npy', path_dir + 'ratios_transport.npy',
                        path_dir + 'ratios_err_transport.npy'])
        dp2 = np.array(
            [path_dir + 'tread_blockade.npy', path_dir + 'ratios_blockade.npy', path_dir + 'ratios_err_blockade.npy'])
        data_1 = load_multiple_data(*dp1)
        data_2 = load_data(*dp2)
        for i in range(len(data_1[1])):

            x_block = data_2[0]#[:20]
            y_block = data_2[1]#[:20]
            y_err_block = data_2[2]#[:20]

            x_transport = data_1[0]#[:20]
            y_transport = data_1[1][i]#, :20]
            y_err_transport = data_1[2][i]#, :20]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plot_data(ax1, x_block, y_block, y_err_block)
            plot_data(ax2, x_transport, y_transport, y_err_transport)
            plt.show()

            prior_pred, trace = fitting_model(x_transport, x_block, y_transport, y_block, *p0)

            y1_prior_samples = prior_pred.prior["y1_pred"].values[0]  # Shape: (1000, len(x1))
            y2_prior_samples = prior_pred.prior["y2_pred"].values[0]  # Shape: (1000, len(x1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            print("Shape of x1:", data_1[0].shape)
            print("Shape of y1_prior_samples:", y1_prior_samples.shape)
            print("Shape of y1_prior_samples[i, :]:", y1_prior_samples[0, :].shape)

            for i in range(5):
                plot_data(ax1, x_transport, y1_prior_samples[i, :])
                plot_data(ax2, x_block, y2_prior_samples[i, :])

            plt.show()

            model_analyz([x_transport, y_transport, y_err_transport],
                         [x_block, y_block, y_err_block], prior_pred, trace)



if __name__ == "__main__":
    p0_0T = [(1.0, 0.3, 1.0, 0.3),
             (0.5, 0.3, 0.5, 0.3),
             (1.5, 0.04),
             (16.0, 1.5),
             (0.2, 0.2)]

    p0_400mT = [(1.0, 0.5, 1.0, 0.5),
                (0.5, 0.5, 0.5, 0.5),
                (1.5, 0.4),
                (3.0, 2.0),
                (0.2, 0.2)]

    main(p0_400mT
         , '/Users/larsm/PycharmProjects/ElHoBlockade/0T_both_dir/', True)
