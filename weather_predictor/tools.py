import numpy as np
import matplotlib.pyplot as plt


def scatter(x, y):
    """
    Make a scatter plot with jitter.
    """
    x_jitter = x + np.random.normal(size=x.size, scale=.5)
    y_jitter = y + np.random.normal(size=y.size, scale=.5)
    plt.plot(
        x_jitter, y_jitter,
        color='black',
        marker='.',
        linestyle='none',
        alpha=.1,
    )
    plt.show()


def find_day_of_year(year, month, day):
    """
    Convert year, month, date to day of the year.
    January 1 = 0

    Parameters
    ----------
    year: int
    month: int
    day: int

    Returns
    -------
    day_of_year: int
    """
    days_per_month = np.array([
        31,  # January
        28,  # February
        31,  # March
        30,  # April
        31,  # May
        30,  # June
        31,  # July
        31,  # August
        30,  # September
        31,  # October
        30,  # November
        31,  # December
    ])
    # For leap years
    if year % 4 == 0:
        days_per_month[1] += 1

    day_of_year = np.sum(np.array(
        days_per_month[:month - 1])) + day - 1
    return day_of_year


def find_autocorr(values, length=100):
    """
    Parameters
    ----------
    values: array of floats
    length: int
        The number of shifts to calculate,
        the maximum offset in number of samples.

    Returns
    -------
    autocorr: array of floats
    """
    autocorr = []
    for shift in range(1, length):
        correlation = np.corrcoef(values[:-shift], values[shift:])[1, 0]
        autocorr.append(correlation)
    return autocorr


