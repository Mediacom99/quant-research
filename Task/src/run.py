# Main file, only this file should be executed from the command line

#TODO add some user input (just clean, do everythig, only train and so on)
#TODO I SHOULD PROBABLY CHECK THE CUMSUM OF EACH STOCK AND SEE IF IT MAKES SENSE TO HAVE THIS PORTFOLIO (OR JUST BUY THE FIRST STOCK)

#IMPLEMENT OPTIMIZATION


#TODO cross_val_score of all the linear models to see which one performs better
#TODO calculate forecasted variance matrix and compare with the one calculated using weights
#TODO once you have weekly data, perform optimization on the weekly data and check the portfolio
#TODO rebalance every week for the last year of data, for each week calculate what you need for the portfolio (return, variance, sharpe, var, svar)
    # (returns and variance are calculated using the true data)
#TODO I should give a weekly and monthly portfolio weight matrix. 
#TODO Should also give results on different timeperiods based on the volatility of the variance of stock indices


#The idea is:
# 1. calculate weights for current week, calculate portfolio weights and use them in that week.
# 2. add that week to the training data and repeat
# 3. do this for the last year kept as training data
# 4. check how the portfolio performs in that week
# 5. Then I can create the portfolio weight matrix for each week in that year (as excel file would be nice)

#Maybe I can also do this in a monthly timeframe
    

# Custom modules
import eda
import clean_data
import model
import pandas as pd
import logging

#Logger configuration
logging.basicConfig(level=logging.DEBUG, filename='minerva-task.log', filemode='w+') # filemode = 'a' for appending to same file

print("Logs are saved in \"minerva-task.log\"")

#Pandas settings
pd.options.mode.copy_on_write = True

# Clean data (this module cleans raw data)
# clean_data.clean_data_run()


# Exploratory data anlysis
# eda.eda_run()


#Model training and weights calculation
model.run(pd.tseries.offsets.BDay(1), divide_years=16)



logging.shutdown()
