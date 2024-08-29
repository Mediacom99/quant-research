
'''
Main file, only this file should be executed from the command line
'''

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

# Filepaths

#Original raw data
rawDataPath = '../raw-data/data.xlsx'

#Cleaned and formatted data
formattedDatPath = '../formatted-data/formatted-data.xlsx'

# Clean data (this module cleans raw data)
# clean_data.clean_data_run(rawDataPath, formattedDatPath)

# Exploratory data anlysis
# eda.eda_run(formattedDatPath, skipAndersonDarling=True)

#Model training and weights calculation
model.tradingModelRun(
          formattedDataPath = formattedDatPath,
          OFFSET = pd.tseries.offsets.BDay(1),
          divide_years = 10,
          print_pca_factor_loadings = False,
          do_cross_validation = False
          )

logging.shutdown()
