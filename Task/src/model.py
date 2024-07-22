"""

This module deals with training the model and forecasting returns based on factors
using PCA on factors and a linear multi-factor model (Linear regression).

"""

from matplotlib import pyplot as plt
import utils
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import eda 

#Calculate PCA over given dataframe, returing the transformed data
#It keeps enough PC to reach a certain variance passed as parameter (between 0 and 1)
def pca_transform_wrapper(factors:pd.DataFrame, desired_var:float = 0.95, print_loadings:bool = False):
    
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
            
    #pca_factors = pd.concat(factor_training, axis=1) #Merge all factors into one bing dataframe        
    
    #Desired percentage of variance to explain with principal components
    desired_var = 0.95
    
    #PCA on Macro and Fundamentals
    pca_factors_macro_fund = pd.concat([factor_training['Macro indices'], factor_training['Fundamentals']], axis=1)
    #print("Factors covariance matrix before PCA:")
    #print(pca_factors.cov())
    
    macro_fund_trans = pca_transform_wrapper(factors=pca_factors_macro_fund, desired_var=desired_var)
    #print("Factors covariance matrix after PCA:")
    #print(utils.clean_cov_matrix(trans, 0.00001))
    
    
    pca_factors_final = pd.concat([
                            factor_training['Rates returns'],
                            factor_training['Forex returns'],
                            factor_training['Commodities returns'],
                            macro_fund_trans], axis=1)
    
    final_pca_trans = pca_transform_wrapper(factors=pca_factors_final, desired_var=desired_var)
    
    print(utils.clean_cov_matrix(final_pca_trans, 0.00001))
    
    
    
    #Regression of returns_training against final_pca_trans
    
    #Create lagged factors dataframe
    factors = final_pca_trans.shift(periods=1, freq='B').dropna()
    
    # I have to remove the first row of returns and the last row of factors (because of the shift)
    factors = factors.drop(factors.index[-1])
    returns = returns_training.drop(returns_training.index[0])

    
    
    
    model = MultiOutputRegressor(LinearRegression())
    a = model.fit(factors, returns)
    
    
    # Get coefficients and intercepts
    coefficients = np.array([estimator.coef_ for estimator in model.estimators_])
    intercepts = np.array([estimator.intercept_ for estimator in model.estimators_])

    # Create a DataFrame of coefficients
    coef_df = pd.DataFrame(coefficients.T, 
                        columns=returns.columns, 
                        index=factors.columns)

    print("\nCoefficients:")
    print(coef_df)

    print("\nIntercepts:")
    print(pd.Series(intercepts, index=returns.columns))
    
    print("Residuals: ")
    osl_residuals = returns - model.predict(factors)
    
    #TODO compare different methods of regression (Elastic Net, OSL, GSL, RandomForest, GradientBoosting)
        
    return