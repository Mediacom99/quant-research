# This module takes raw data (from excel file) and formats it correctly in order to use it for further statistical purposes.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model  import LinearRegression


data = pd.read_excel("data.xlsx", sheet_name=[0,1,2,3,4,5])
returns = data[0];

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

    print("Zeroes:", indices_to_change.size)


    # Loop through indices of zeros to change
    for idx in indices_to_change:

        # Find date of 5 days before and after this particular zero date
        start_date = idx - pd.DateOffset(days=days_before)
        end_date = idx + pd.DateOffset(days=days_after)

        # Extract values from timeframe
        data_range = series.loc[start_date:end_date]

        # Calculate mean 
        mean_return = data_range.mean()

        # Update the Series with the predicted value for zeros
        series.loc[idx] = mean_return


returns.to_excel('RendimentiFormat.xlsx')
