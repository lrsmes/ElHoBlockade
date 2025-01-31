import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az


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
            az.plot_dist(trace[param].values.flatten(), ax=axes[i], kind="kde", label=param)
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


def fitting_model(x1, x2, y1, y2):
    with pm.Model() as model:
        A1 = pm.Normal("A1", sigma=1.0, mu=0.3)
        A2 = pm.Normal("A2", sigma=1.0, mu=0.3)

        B1 = pm.Normal("B1", sigma=.5, mu=0.3)
        B2 = pm.Normal("B2", sigma=.5, mu=0.4)

        lamda1 = pm.Weibull("lamda1", alpha=1.5, beta=0.04)
        r = pm.Normal('r', mu=10, sigma=10)
        lamda2 = pm.Deterministic('lamda2', r * lamda1)

        y1_pred = pm.Deterministic("y1_pred", A1 * pm.math.exp(-1 * x1 / lamda1) + B1)
        y2_pred = pm.Deterministic("y2_pred", A2 * pm.math.exp(-1 * x2 / lamda2) + B2)

        sigma1 = pm.HalfNormal("sigma1", sigma=0.1)
        sigma2 = pm.HalfNormal("sigma2", sigma=0.2)
        y1_obs = pm.Normal("y1_obs", mu=y1_pred, sigma=sigma1, observed=y1)
        y2_obs = pm.Normal("y2_obs", mu=y2_pred, sigma=sigma2, observed=y2)

        prior_pred = pm.sample_prior_predictive(samples=10000)

        trace = pm.sample(4000, tune=4000, return_inferencedata=True, target_accept=0.99, chains=4)
    return prior_pred, trace


def main():
    path_dir = '/Users/hubert.D/Documents/0T_ratios/'
    dp1 = np.array([path_dir + 'tread_transport.npy', path_dir + 'ratios_transport.npy', path_dir + 'ratios_err_transport.npy'])
    dp2 = np.array(
        [path_dir + 'tread_blockade.npy', path_dir + 'ratios_blockade.npy', path_dir + 'ratios_err_blockade.npy'])
    data_1 = load_data(*dp1)
    data_2 = load_data(*dp2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_data(ax1, data_1[0], data_1[1], data_1[2])
    plot_data(ax2, data_2[0], data_2[1], data_2[2])
    plt.show()
    prior_pred, trace = fitting_model(data_1[0], data_2[0], data_1[1], data_2[1])

    y1_prior_samples = prior_pred.prior["y1_pred"].values[0]  # Shape: (1000, len(x1))
    y2_prior_samples = prior_pred.prior["y2_pred"].values[0]  # Shape: (1000, len(x1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    print("Shape of x1:", data_1[0].shape)
    print("Shape of y1_prior_samples:", y1_prior_samples.shape)
    print("Shape of y1_prior_samples[i, :]:", y1_prior_samples[0, :].shape)

    for i in range(10):
        plot_data(ax1, data_1[0], y1_prior_samples[i, :])
        plot_data(ax2, data_2[0], y2_prior_samples[i, :])

    plt.show()

    # Extract mean values of each parameter
    posterior_means = az.summary(trace, kind="stats")["mean"]
    print("\nMean Posterior Estimates:")
    print(posterior_means)

    # Plot prior distributions
    plot_parameter_distributions(prior_pred.prior, title="Prior Distributions")

    # Plot posterior distributions
    plot_parameter_distributions(trace.posterior, title="Posterior Distributions")

    # Convert to a dictionary
    posterior_dict = posterior_means.to_dict()
    # Convert to a list (sorted by parameter name)
    posterior_list = list(posterior_dict.values())
    print(posterior_list[:8])


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_data(ax1, data_1[0], data_1[1], data_1[2])
    ax1.plot(np.arange(0.0, 0.15, 0.001), model(np.arange(0.0, 0.15, 0.001), posterior_means["A1"], posterior_means["B1"], posterior_means["lamda1"]))
    plot_data(ax2, data_2[0], data_2[1], data_2[2])
    ax2.plot(np.arange(0.0, 1.35, 0.001), model(np.arange(0.0, 1.35, 0.001), posterior_means["A2"], posterior_means["B2"], posterior_means["lamda2"]))
    plt.show()




if __name__ == "__main__":
    main()
