# Main file, only this file should be executed
#TODO add some user input (just clean, do everythig, only train and so on)

# OSL : -1.2259240622777985
# Huber: broken
# SDG regressor: -1.1186912582156785

# Custom modules
import eda
import clean_data
import model
import pandas as pd

#Pandas settings
pd.options.mode.copy_on_write = True

# Clean data (this module cleans raw data)
#clean_data.clean_data_run()

#eda.eda_run()

#Model training and weights calculation
model.model_train()