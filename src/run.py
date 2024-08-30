

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
logging.basicConfig(level=logging.CRITICAL, filename='minerva-task.log', filemode='w+')

print("Logs are saved in \"minerva-task.log\"")


#Pandas settings
pd.options.mode.copy_on_write = True

# Filepaths

#Original raw data
rawDataPath = '../raw-data/data.xlsx'

#Cleaned and formatted data
formattedDatPath = '../formatted-data/formatted-data.xlsx'

# Clean data (this module cleans raw data)
#clean_data.cleanDataRun(rawDataPath, formattedDatPath)

# Exploratory data anlysis
#eda.edaRun(formattedDatPath, skipAndersonDarling=True)

# Run the trading model
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
          divide_years = 16,
          print_pca_factor_loadings = False,
          do_cross_validation = False
          )

logging.shutdown()
