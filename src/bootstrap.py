import numpy as np


def get_bootstrap_ci(df, num_experiments=1000, alpha=0.05):
    """Get bootstrap confidence interval and median for the input dataframe.

    Args:
        df (DataFrame): couriers distribution by hours.
        num_experiments (int): num of experiments repeats.
        alpha (float): statistical significance.
    Returns:
        Computed confidence interval and estimated median.
    """
    mean_data = []
    for i in range(num_experiments):
        sample = df.sample(frac=0.5, replace=True)
        mean = sample.mean()
        mean_data.append(mean)

    ci = np.percentile(mean_data, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    median = np.percentile(mean_data, 0.5)
    return ci, median
