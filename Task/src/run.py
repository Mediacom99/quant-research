# Main file, only this file should be executed
#TODO add some user input (just clean, do everythig, only train and so on)

# Custom modules
import eda
import clean_data
import model

# Clean data (this module cleans raw data)
clean_data.clean_data_run()

#Exploratory data anlysis (from this point on we use cleaned data saved in FormattedData folder)
#eda.eda_run()

#Model training and weights calculation
model.model_train()