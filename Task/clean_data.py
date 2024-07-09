# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.

import pandas as pd



def mean_clean(returns):
    # Set timestamps as dataframe index (year-month-day)
    returns = returns.set_index('Date')

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


def main():
    data = pd.read_excel("data.xlsx", sheet_name=[0,1,2,3,4,5])
    returns = data[0];
    fund = data[5];

    fund.set_index('Dates', inplace=True)

    fund_daily = fund.resample('D').interpolate()

    #print(fund_daily.describe())

    returns = mean_clean(returns)
    returns.to_excel('RendimentiFormat.xlsx')
    print(returns.describe())
    ## Remove weekends and CHECK WHAT INTERPOLATE DOES, ARE THERE NULL VALUES ?
    return


main()

