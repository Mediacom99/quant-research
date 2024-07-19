"""

This module deals with training the model and forecasting returns based on factors
using PCA on factors and a linear multi-factor model (Linear regression).

"""

import utils
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Calculate PCA over given dataframe, returing the transformed data
#It keeps enough PC to reach a certain variance passed as parameter (between 0 and 1)
def pca_transform_wrapper(factors:pd.DataFrame, desired_var:float = 0.8, print_loadings:bool = False):
    
    pca = PCA()
    feature_names = factors.columns
    pca.fit_transform(factors)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) #Cumulative sum of variance ratios
    
    
    #Choose number of components
    n_components = np.argmax(cumulative_variance_ratio >= desired_var) + 1
    print(f"Number of components explaining {desired_var*100}% of variance: {n_components}")
    
    pca = PCA(n_components=n_components)
    factor_training = pca.fit_transform(factors) #New factor_training dataset after PCA
    
    #This dataframe, if made up of all the PC, is the inverse (or transpose since orthogonal) of the change of basis matrix
    #between the original space and the space where the covariance matrix between factors is diagonal. Basically you can see
    # how much each factors contributes to each PC.
    loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i}' for i in range(n_components)],
    index=feature_names
    )
    if(print_loadings):
        print("Loadings:")
        print(loadings)
    
    factor_df = pd.DataFrame(factor_training, columns=[f'PC{i}' for i in range(n_components)], index=factors.index)
    
    return factor_df    



def model_train():
    
    data = utils.get_data_from_excel('./FormattedData/formatted-data.xlsx')
    
    #Normalizing data to have zero mean and unit variance 
    #TODO I could also normalize first stocks and then normalize the rest with respect to stock    
    for df in data:
        data[df] = utils.normalize_dataframe(data[df])
    
    
    # Divide dataset into training and testing datasets
    factor_training = {}
    factor_testing = {}
    returns_training, returns_testing = utils.divide_df_lastyear(data['Stock returns'])
    for df in data:
        if(df != 'Stock returns'):
            factor_training[df], factor_testing[df] = utils.divide_df_lastyear(data[df])
            
    pca_factors = pd.concat(factor_training, axis=1) #Merge all factors into one bing dataframe        
    
    #Perform PCA
    #Perform Linear Regression
    #Forecast returns into testing
    print("Factors covariance matrix before PCA:")
    print(pca_factors.cov())
    
    trans = pca_transform_wrapper(factors=pca_factors, desired_var=1)
    
    print("Factors covariance matrix after PCA:")
    print(utils.clean_cov_matrix(trans, 0.00001))
        

    
    return