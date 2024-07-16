"""

This module deals with training the model and forecasting returns based on factors
using PCA+APT+LinearRegression or with the training of a LSTM Neural Network + LRP

"""

import utils

def model_train():
    
    data = utils.get_data_from_excel('./FormattedData/formatted-data.xlsx')
    
    #Normalizing data to have zero mean and unit variance 
    #TODO I could also normalize first stocks and then normalize the rest with respect to stock    
    for df in data:
        data[df] = utils.normalize_dataframe(data[df])
    
    
    # Divide dataset into training and testing datasets
    data_training = {}
    data_testing = {}
    for df in data:
        data_training[df], data_testing[df] = utils.divide_dataframe_lastyear(data[df])
        
    #Now data_training and data_testing are collections just like data, one with training data 
    # and the other with testing data
    
    #Perform PCA
    #Perform Linear Regression
    #Forecast returns into testing
    
    return