# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.
# This module should be used once to clean the raw data, it will output an execl file containing all the cleaned data,
# properly formatted into daily data.
# The data is cleaned by replacing every 0 (zero) and NaN with a moving average at their original timeframe,
# after that everything is reindexed into a daily timeframe.

import pandas as pd
import eda
import utils 


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

def clean_data_run():
    # Xls is the excel file with different sheets, each one will become a certain dataframe
    data = utils.get_data_from_excel('./data/data.xlsx')
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


    print("INFO: writing xlsx file...")
    #Print each cleaned dataframe into its own excel file
    with pd.ExcelWriter('./FormattedData/formatted-data.xlsx') as writer:
        returns.to_excel(writer, sheet_name='Stock returns')
        rates.to_excel(writer, sheet_name='Rates returns')
        forex.to_excel(writer, sheet_name='Forex returns')
        commod.to_excel(writer, sheet_name='Commodities returns')
        macro_daily.to_excel(writer, sheet_name='Macro indices')        
        fund_daily.to_excel(writer, sheet_name='Fundamentals')

    return



clean_data_run()
print("Dataset cleaning finished successfully!")
print("")


