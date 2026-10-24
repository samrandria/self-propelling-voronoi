"""
Test for the gaussian white noise process
"""

from matplotlib import pyplot as plt 
import numpy as np 
from flocking_confluent_marchetti import gaussian_white_noise_process
from pandas.tools.plotting import autocorrelation_plot

if __name__ == '__main__':

    random_samples = gaussian_white_noise_process(0,1,10000)
    # Plot the different samples
    plt.figure("Normal plot")
    plt.plot(random_samples)

    # Plot an histogram to show Gaussian distribution
    plt.figure("Histogram")
    n, bins, patches = plt.hist(random_samples, 'auto')

    # Plot the autocorrelation to make sure the processes are actually decorrelated
    plt.figure("Autocorrelation")
    plt.acorr(random_samples)

    plt.figure("Pandas autocorrelation")
    autocorrelation_plot(random_samples) 

    plt.show()