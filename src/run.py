

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

# CLEAN DATA (this module cleans raw data)
# clean_data.cleanDataRun(rawDataPath, formattedDatPath)

# EXPLORATORY DATA ANLYSIS
# eda.edaRun(formattedDatPath, skipAndersonDarling=True)

# RUN THE TRADING MODEL
"""
Valid OFFSET values:
- X business years: ofs.BYearEnd(X)
- X business months: ofs.BMonthEnd(X)
- X business weeks: ofs.BDay(5 * X)
- X business days: ofs.BDay(X)
"""

model.tradingModelRun(
          formattedDataPath = formattedDatPath,
          OFFSET = ofs.BYearEnd(1),
          divide_years = 1, #Max is 16
          print_pca_factor_loadings = False,
          do_cross_validation = False,
          portfolio_matrix_filename = '../portfolio-matrices/bohbohboh.xlsx'
          )

logging.shutdown()
