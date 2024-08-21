""" 
 This module takes raw data (from the excel file data/data.xlsx) and cleans it for further statistical calculations.
 This module should be used once to clean the raw data, it will output an execl file containing all the cleaned data,
 properly formatted into daily data. This file is FormattedData/formatted-data.xlsx
 The data is cleaned by replacing every 0 (zero) and NaN with a forward and backward fill.
 Then a rolling average is applied only to the replaced data.
 Everything that is not comprised of daily data is resampled into business days daily data using forward fill.
 """
from matplotlib import pyplot as plt
import pandas as pd
import utils 
import numpy as np
import logging 


log = logging.getLogger('clean_data')


def fill_mean(data, window = '5D'):
    """
    Forward and backword filling of zeros and NaN values. Apply rolling average
    only over the previously filled values.
    
    Args:
    data : dataframe to clean
    window : time window to use for rolling average, default is 5 days (check pandas offset aliases)
    
    Returns: cleaned dataframe as a new dataframe
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
    """
    Main function for this module. It is called once in the entry point run.py. 
    It loads data from data.xlsx, cleans, resamples and check for residuals nans.
    Writes a xlsx file containing every dataframe in a different sheet.
    """
    log.info("CLEAN_DATA.PY STARTING")
    log.info("loading raw data.xlsx file...")
    data = utils.get_data_from_excel('./data/data.xlsx')
    
    #Log transformation of macro indices and fundamentals
    data['Macroeconomics'] = utils.log_transform(data['Macroeconomics'])
    data['Fondamentali Indici Azionari'] = utils.log_transform(data['Fondamentali Indici Azionari'])
    
    ## DATA CLEANING
    log.info("cleaning dataframes...")
    for df in data:
        data[df] = fill_mean(data[df]) 


    # Resampling monthly and quarterly data into daily data using forward fill    
    log.info("starting resampling into daily timeframe...") 
    macro_daily = data['Macroeconomics'].resample('B').ffill()
    fund_daily = data['Fondamentali Indici Azionari'].resample('B').ffill()
    
    #Check for nan values after cleaning
    utils.count_nans(data)


    #Divide each sheet into its own dataframe
    returns =   data['Rendimenti Indici Azionari'] 
    rates   =   data['Tassi'] 
    forex   =   data['Forex'] 
    commod  =   data['Commodities']
    

    log.info("writing formatted-data xlsx file...")
    #Print each cleaned dataframe into its own sheet in the same excel file
    with pd.ExcelWriter('./FormattedData/formatted-data.xlsx') as writer:
        returns.to_excel(writer, sheet_name='Stock returns')
        rates.to_excel(writer, sheet_name='Rates returns')
        forex.to_excel(writer, sheet_name='Forex returns')
        commod.to_excel(writer, sheet_name='Commodities returns')
        macro_daily.to_excel(writer, sheet_name='Macro indices')        
        fund_daily.to_excel(writer, sheet_name='Fundamentals')
    
    log.info("dataset cleaned successfully!")
    return
