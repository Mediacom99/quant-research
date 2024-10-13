'''
Main file, only this file should be executed from the command line
'''

import eda
import clean_data
import model
import pandas as pd
import logging
import pandas.tseries.offsets as ofs

#Logger configuration
logging.basicConfig(level=logging.INFO, filename='minerva-task.log', filemode='w+')

#Pandas settings
pd.options.mode.copy_on_write = True

print("More info can be found in the log file: \"minerva-task.log\"")

#Original raw data
rawDataPath = '../raw-data/data.xlsx'

#Cleaned and formatted data
formattedDatPath = '../formatted-data/formatted-data.xlsx'

# Uncomment to clean data.xlsx and write formatted-data.xlsx
clean_data.cleanDataRun(rawDataPath, formattedDatPath)

# Uncomment to perform exploratory data analysis.
eda.edaRun(formattedDatPath, doAndersonDarling=False) #Set to True for Anderson-Darling test


"""
Valid OFFSET/STOP_OFFSET values:
- X business years: ofs.BYearEnd(X)
- X business months: ofs.BMonthEnd(X)
- X business weeks: ofs.BDay(5 * X)
- X business days: ofs.BDay(X)
"""

# Comment/Uncomment to run the trading model with rolling window
model.tradingModelRun(
          formattedDataPath = formattedDatPath,
          OFFSET = ofs.BDay(5),
          divide_years = 16, #Max is 16
          print_pca_factor_loadings = False, #Set to True for pca factor loadings
          do_cross_validation = False, #Set True to perform cross validation on each run (use BYearEnd(1) offset)
          portfolio_matrix_filename = '../portfolio-matrices/portfolio-matrix-temp.xlsx',
          STOP_OFFSET = ofs.BYearEnd(0), #To stop the testing before end of historical data
          )

logging.shutdown()
