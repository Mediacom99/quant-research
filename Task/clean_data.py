# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from scipy.stats import ttest_rel, norm
from scipy.optimize import curve_fit

def mean_clean(returns):

    # Define the number of days before and after for calculating mean
    days_before = 5
    days_after = 5

    ## Change dataframe so that each zero gets replaced by the mean of the previous and next 5 days

    for cols in returns.columns:

        series = returns[cols]
        # Find indices of zeros
        indices_to_change = series.index[series == 0]

        print("Number of zeroes:", indices_to_change.size)


        # Loop through indices of zeros to change
        for idx in indices_to_change:

            # Find date of 5 days before and after this particular zero date
            start_date = idx - pd.DateOffset(days=days_before)
            end_date = idx + pd.DateOffset(days=days_after)

            # Extract values from timeframe
            data_range = series.loc[start_date:end_date]

            # Calculate mean and update series
            series.loc[idx] = data_range.mean()

    return returns

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))



#TODO fundamentals: Remove weekends and CHECK WHAT INTERPOLATE DOES, ARE THERE NULL VALUES ?
#TODO Also check higher moments between our data and normal pdf 
def main():
    data = pd.read_excel("data.xlsx", sheet_name=[0,1,2,3,4,5])
    returns = data[0];
    fund = data[5];

    #Set date column as index
    returns = returns.set_index('Date')
    fund = fund.set_index('Dates')

    fund_daily = fund.resample('D').interpolate()

    #print(fund_daily.describe())

    returns = mean_clean(returns)
    returns.to_excel('RendimentiFormat.xlsx')
    


    series = returns['Indice Azionario Paese 5']
    
    
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


main()

