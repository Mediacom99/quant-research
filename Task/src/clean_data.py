# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.
# This module should be used once to clean the raw data, it will output an execl file containing all the cleaned data,
# properly formatted into daily data.
# The data is cleaned by replacing every 0 (zero) and NaN with a moving average at their original timeframe,
# after that everything is reindexed into a daily timeframe.

import pandas as pd
import utils 
import numpy as np


def fill_mean(data, window = '5D'):
    """
    Forward and backword filling of zeros and NaN values. Apply rolling average
    only over the previously filled values.
    
    Parameters:
    data : dataframe to clean
    window : time window to use for rolling average, default is 10 days (check pandas offset aliases)
    
    Output: cleaned dataframe as a new dataframe
    This function does not change the passed dataframe 
    """
    
    #Replace zeroes with NaN
    df = data.copy()
    df = df.replace(to_replace = 0, value = np.nan)
    for cols_name, series in df.items():
        # Mask to keep record of position of NaN values
        nan_mask = series.isna()
        series = series.ffill()
        series = series.bfill()
        #Apply moving average to filled values
        df.loc[nan_mask, cols_name] = series.rolling(window = window).mean()
                
    return df

def clean_data_run():
    # Xls is the excel file with different sheets, each one will become a certain dataframe
    data = utils.get_data_from_excel('./data/data.xlsx')
    
    ## DATA CLEANING
    print("INFO: cleaning dataframes...")
    """ returns = fill_mean(returns)
    rates = fill_mean(rates)
    macro = fill_mean(macro)
    forex = fill_mean(forex)
    commod = fill_mean(commod)
    fund = fill_mean(fund) """
    for df in data:
        data[df] = fill_mean(data[df]) 


    # Change monthly and quarterly data into daily data using forward fill    
    print("INFO: starting resampling into daily timeframe...") 
    macro_daily = data['Macroeconomics'].resample('D').ffill()
    fund_daily = data['Fondamentali Indici Azionari'].resample('D').ffill()
    
    #Check for nan values after cleaning
    utils.count_nans(data)


    #Divide each sheet into its own dataframe
    returns =   data['Rendimenti Indici Azionari'] 
    rates   =   data['Tassi'] 
    forex   =   data['Forex'] 
    commod  =   data['Commodities']
    

    print("INFO: writing xlsx file...")
    #Print each cleaned dataframe into its own excel file
    with pd.ExcelWriter('./FormattedData/formatted-data.xlsx') as writer:
        returns.to_excel(writer, sheet_name='Stock returns')
        rates.to_excel(writer, sheet_name='Rates returns')
        forex.to_excel(writer, sheet_name='Forex returns')
        commod.to_excel(writer, sheet_name='Commodities returns')
        macro_daily.to_excel(writer, sheet_name='Macro indices')        
        fund_daily.to_excel(writer, sheet_name='Fundamentals')
    
    print("Dataset cleaning finished successfully!")
    
    return
