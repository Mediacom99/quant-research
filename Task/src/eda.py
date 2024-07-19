# Exploratory Data Analysis of cleaned data

import pandas as pd
from scipy import stats as st
import numpy as np
from matplotlib import pyplot as plt

import utils

# Fit series against normal distribution. Provide mean, variance, t-test and p-value. (over N simulations)
def normal_fit_test(series: pd.Series):
    
    print("INFO: starting goodness of fit test")
    data = series.values
    
    rng = np.random.default_rng()
    #Using scipy.stats.goodness_of_fit
    #known_params = {'loc' : data.mean(), 'scale' : data.std()}
    result = st.goodness_of_fit(st.norm, data, statistic='ad', random_state=rng)

    return result

#TODO probably wrong, should check Mardia's test
def multi_norm_fit_test(data: pd.DataFrame):
    
    mean = data.mean()
    cov = data.cov()
    N = len(mean)
    
    #Generate data with mean and cov of data
    data = st.multivariate_normal.rvs(mean = mean, cov = cov)
    
    #Calculate Mahalanobis distances (multivariate distance like q-squared)
    mahab_dist = np.sqrt( ((data - mean) @ np.linalg.inv(cov)) ** 2) # @ is matrix multiplication operator in numpy
    
    #Calculate p-value using fact that mahab_dist follows a chi-squared distribution
    chi2_stat = np.sum(mahab_dist ** 2)
    pvalue = 1 - st.chi2.cdf(chi2_stat, N)
    #print(f"P-value of normal multivariate fitting: {pvalue}")
    
    return pvalue

def shapiro_wilk_norm(data):
    
    #for cols_name, series in data.items():
    #    res, pvalue = st.shapiro(series.values)
    #    print(f"P-value of normal fit for {cols_name}: {pvalue}")
    
    res, p = st.shapiro(data['Indice Azionario Paese 1'].values)
    return p



#TODO calculate kurtosis and skewness
#TODO plot variance over different time periods (each week, each month and each year)
#TODO visualize stock returns monthly over the whole timeframe
#TODO LAG_PLOT FOR RANDOMNESS OF TIME SERIES
def  eda_run():

    data = utils.get_data_from_excel('./FormattedData/formatted-data.xlsx')
    returns_norm = utils.normalize_dataframe(data['Stock returns'])
    print(returns_norm.head())

    """ 
    for cols_name, series in returns_norm.items():
        res = normal_fit_test(series)
        print(cols_name, ":")
        print(f"(p-value, statistic) -> ({res.pvalue},{res.statistic})")
        print(f"Estimated params (loc, scale): {res.fit_result.params.loc}, {res.fit_result.params.scale}")
        print("")
     """
     
    
    sum = 0
    sum2 = 0
    K = 10000
    #for i in range(K):
        #sum += multi_norm_fit_test(returns_norm)
        #sum2 += shapiro_wilk_norm(returns_norm)
    print(f"Average p-value multivariate distribution: {sum/K}")
    print(f"Average p-value distribution: {sum2/K}")
    return