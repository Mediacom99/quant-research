# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.
# This module should be used once to clean the raw data, it will output an execl file containing all the cleaned data,
# properly formatted into daily data.
# The data is cleaned by replacing every 0 (zero) and NaN with a moving average at their original timeframe,
# after that everything is reindexed into a daily timeframe.

import pandas as pd

#TODO There is definitely a better way to do this.
## Takes a date indexed dataframe as input, finds all dates corresponding to either zeros or null values and it replaces them with a moving-average of period 5 days
## period is the number of days (both before and after) to use for the moving-average. Standard is 5 days before and after. Timetp specifies the timestep to use
## when calculating time offset for moving average, for example timetp='months' would calculate moving average every 'period' months.
def mean_clean(returns, period = 5, timetp='days'):

    # Loop over series of dataframe
    for cols_name, series in returns.items():

        # Find indices of zeros and null values
        indices_to_change = series[series == 0].index
        null_values = series[series.isnull()].index

        print(cols_name)
        print("Zeroes: ", indices_to_change.size)
        print("NaN: ", null_values.size)

        #timedelta = pd.Timedelta(period, timetp)

        if timetp == 'days':
            timeoffset = pd.DateOffset(days = period)            
        if timetp == 'months':
            timeoffset = pd.DateOffset(months = period)
                
        
        
        # Loop through indices of zeros to change
        for idx in indices_to_change:

            # Find date of 'period' days before and after this particular zero date
            start_date = idx - timeoffset;
            end_date = idx + timeoffset;
            # Extract values from timeframe
            data_range = series.loc[start_date:end_date]
            # Calculate mean and update series
            series.loc[idx] = data_range.mean()

        for ndx in null_values:
            start_date = ndx - timeoffset;
            end_date = ndx + timeoffset;
            data_range = series.loc[start_date:end_date]
            if data_range.isnull().all(): # This is so that there are not more NaN values than the period of the mov average
                print('For column \'{}\' use a bigger period, the moving-average is full of NaN values!'.format(cols_name))
                print("Start date: ", start_date)
                print("End date: ", end_date)
                print("Stopping replacing NaN values!")
                break;
            series.loc[ndx] = data_range.mean()

        print("")

    return returns

def clean():
    # Xls is the excel file with different sheets, each one will become a certain dataframe
    xls = pd.ExcelFile("data.xlsx")
    sheet_names = xls.sheet_names
    data = {}
    
    for sheet_name in sheet_names:
        data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
        
    for df in data:
        data[df]['Date'] = pd.to_datetime(data[df]['Date']) #Format Date to datetime format
        data[df].set_index('Date', inplace=True) #Set Date column as index

    #Divide each sheet into its own dataframe
    returns =   data['Rendimenti Indici Azionari'] 
    rates   =   data['Tassi'] 
    macro   =   data['Macroeconomics'] 
    forex   =   data['Forex'] 
    commod  =   data['Commodities'] 
    fund    =   data['Fondamentali Indici Azionari']
    

    ## DATA CLEANING
    print("INFO: starting cleaning dataframes...")
    returns = mean_clean(returns)
    rates = mean_clean(rates)
    macro = mean_clean(macro, 17, 'months')
    forex = mean_clean(forex)
    commod = mean_clean(commod)
    fund = mean_clean(fund, 3, 'months') #you can change 6 with any multiple of 3 (data is quarterly)    

    print("INFO: starting reindexing into daily timeframe...")
    # Date range to reindex monthly and quarterly data into daily data
    business_days = pd.date_range(start=returns.index.min(), end=returns.index.max(), freq='B')
    macro_daily =  macro.reindex(business_days, method='pad')
    fund_daily = fund.reindex(business_days, method='pad')

    print("INFO: writing xlsx files...")
    #Print each cleaned dataframe into its own excel file
    returns.to_excel('FormattedData/ReturnsFormat.xlsx')
    rates.to_excel('FormattedData/RatesFormat.xlsx')
    forex.to_excel('FormattedData/ForexFormat.xlsx')
    commod.to_excel('FormattedData/CommoditiesFormat.xlsx')
    macro_daily.to_excel('FormattedData/MacroFormat.xlsx')        
    fund_daily.to_excel('FormattedData/FundamentalsFormat.xlsx')

    
    return



clean()

