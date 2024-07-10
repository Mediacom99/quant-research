# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.
# This module should be used once to clean the raw data, it will output an execl file containing all the cleaned data,
# properly formatted

import pandas as pd
import eda #custom exploratory data anlysis module

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
    
    return


main()

