# Main file, only this file should be executed from the command line

#TODO add some user input (just clean, do everythig, only train and so on)

# Custom modules
import eda
import clean_data
import model
import pandas as pd
import logging

logging.basicConfig(level=logging.CRITICAL, filename='minerva-task.log', filemode='w+') # filemode = 'a' for appending to same file


#Pandas settings
pd.options.mode.copy_on_write = True

# Clean data (this module cleans raw data)
#clean_data.clean_data_run()

#eda.eda_run()



#Model training and weights calculation
model.run()

logging.log(logging.CRITICAL, "--------------------------------------------------------------------------------------------------------------------------------------\n\n")

logging.shutdown()