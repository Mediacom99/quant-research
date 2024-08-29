# Exploratory Data Analysis of cleaned data

import pandas as pd
from scipy import stats as st
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import logging as log
import utils

logger = log.getLogger('eda')


def normalFitTest(series: pd.Series, statistic:str):

    logger.info("starting goodness of fit test")
    data = series.values
    rng = np.random.default_rng()

    #known_params = {'loc' : data.mean(), 'scale' : data.std()}
    result = st.goodness_of_fit(st.norm, data, statistic=statistic, random_state=rng)

    return result


def plotWeeklyStd(df:pd.DataFrame):
    # Calculate weekly variance
    weekly_var = df.resample('W').std()

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    for i, column in enumerate(df.columns):
        axs[i].plot(weekly_var.index, weekly_var[column])
        axs[i].set_title(f'{column} Weekly Standard Deviation')
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('Std')

    # Remove the unused subplot
    fig.delaxes(axs[5])

    plt.tight_layout()
    plt.show()
    return

def correlationAnalysis(df:pd.DataFrame):

    corr_matrix = df.corr(method='pearson')
    plt.figure(figsize=(20, 16))

    #Create heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='plasma', robust=True)

    plt.title('Correlation Matrix of Factors and Stock Returns')
    plt.tight_layout()
    plt.show()

    return corr_matrix


def plotCumReturns(df):

    # Set up the plot
    plt.figure(figsize=(15, 8))

    cum_returns = df.cumsum()

    for column in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[column], label=column)

    plt.title('Cumulative Log Returns of Stocks Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Returns')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Improve the x-axis date formatting
    plt.gcf().autofmt_xdate()

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def  edaRun(formattedDataPath: str, skipAndersonDarling: bool):

    data = utils.get_data_from_excel(formattedDataPath)
    returns_norm = utils.normalize_dataframe(data['Stock returns'])


    for cols_name, series in returns_norm.items():
        print(cols_name, ":")
        print("\tMean: ", series.mean())
        print("\tStd: ", series.std())
        print("\tSkewness", series.skew())
        print("\tExcess kurtosis: ", series.kurtosis())

        if(skipAndersonDarling == False):
            print("\tAnderson-Darling:")
            result_ad = normalFitTest(series, 'ad')
            print("\t\tP-value: ", result_ad.pvalue)

    utils.fiveFigurePlot(returns_norm)

    plotWeeklyStd(returns_norm)

    #Correlation Analysis
    stock_factors_df = pd.concat(data, axis=1)
    corr_matrix = correlationAnalysis(stock_factors_df)
    print("Correlation matrix of the entire dataset:")
    print(corr_matrix)

    print("Stocks cumulative returns:")
    plotCumReturns(data['Stock returns'])

    return
