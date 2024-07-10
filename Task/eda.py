# Exploratory Data Analysis of cleaned data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from scipy.stats import ttest_rel, norm
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))

# Take a pandas series as input, scale data, fit it with normalized gaussian, calculate p-value and other statistical 
# indices to determine the goodness of the fit, lastly plot the gaussian and the data on an histogram.
def gaussian_fitting(series):
    # Scale the data to have mean 0 and std 1
    scaler = StandardScaler()
    scaled_data = pd.Series(scaler.fit_transform(series.values.reshape(-1,1)).flatten())
   

    ## CURVE FIT
    hist, bin_edges = np.histogram(scaled_data, bins=round(np.sqrt(scaled_data.size)), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    params, cov = curve_fit(gaussian, bin_centers, hist, p0=[0, 1, 1])

    print(f"Mean: {params[0]}, Standard Deviation: {params[1]}, Amplitude: {params[2]}")
    #print(cov)

    mu = params[0]
    sigma = params[1]
     

    ## TESTING FIT
    fitted_samples = np.random.normal(loc=mu, scale=sigma, size=scaled_data.size)
    t_statistic, p_value = ttest_rel(scaled_data, fitted_samples)

    print(f"t-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value*100:.2f}%") 
    print(f"Cohen's d: ", mu/sigma)
    print(f"Mean Squared Error: ", np.mean((fitted_samples - scaled_data)**2))


    scaled_data.plot(kind='hist', bins = 500, density=True)
    # Plot fitted Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 5000)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k',linewidth=2)
    plt.tight_layout()
    plt.show() 

    return
