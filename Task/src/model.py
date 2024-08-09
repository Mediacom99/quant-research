
"""

This module deals with training the model and forecasting returns based on factors
using PCA on factors and a linear multi-factor model.

"""

from matplotlib import pyplot as plt
import utils
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, SGDRegressor, ElasticNet, LassoLarsIC, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor

import eda 

#Calculate PCA over given dataframe, returing the transformed data
#It keeps enough PC to reach a certain variance passed as parameter (between 0 and 1)
def pca_transform_wrapper(factors:pd.DataFrame, n_components = 5, print_loadings:bool = False):
    
    feature_names = factors.columns
    pca = PCA(n_components=n_components)
    factor_transformed = pca.fit_transform(factors) #New factor_training dataset after PCA
    
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
    
    factor_df = pd.DataFrame(factor_transformed, columns=[f'PC{i}' for i in range(n_components)], index=factors.index)
    
    return factor_df    

#Linear regression of Y against X
# Y -> stock returns
# X -> pca-factors-lagged (1-d)
# SGDRegressor(loss='huber', shuffle=False)
def linear_regression_model_train(Y, X, model = SGDRegressor(loss='huber', shuffle=False), predict_data = pd.DataFrame()):
     
    
    print(f"INFO: current selected module is: {model}")
    model = MultiOutputRegressor(model)
    
    print(f"INFO: model {model} is calculating weights:")
    model.fit(X, Y)
    
    # Get coefficients and intercepts
    coefficients = np.array([estimator.coef_ for estimator in model.estimators_])
    intercepts = np.array([estimator.intercept_ for estimator in model.estimators_])
    
    
    #FIXME THIS SHIT MIGHT BE WRONG, SHOULD BE (NUM_STOCKS, NUM_FACTORS)
    # Create a DataFrame of coefficients (factors exposures)
    coef_df = pd.DataFrame(coefficients.T, 
                        columns=Y.columns, 
                        index=X.columns)


    #FIXME probably this one is wrong too
    intercepts = pd.DataFrame(intercepts, columns=['Intercepts'], index = Y.columns)

    
    #Instance of timeSeriesSplit class
    #res = cross_val_score(model, X = X,  y = Y,  cv = TimeSeriesSplit(n_splits = 5), error_score='raise', scoring=None)
    # print("Array of scores of the estimator for each run in the cross validation")
    # print(res)
    #print("Cross validation score mean: ", res.mean())
    
    y_pred = model.predict(X)
    residuals = (Y - y_pred)
    print("Residuals mean:")
    print(residuals.mean())
    print("Residuals std:") 
    print(residuals.std())
    
    if (predict_data.empty):
        print("INFO: no predict_data provided. Forecasted data will be returned empty.")
        forecast = []
    else:
        forecast = model.predict(predict_data)
        forecast = pd.DataFrame(forecast, columns = Y.columns)
    
    # coef_df are the weights of the regression, forecast is the data forecasted using those weights given a dataframe of test data
    return (residuals, coef_df, intercepts, forecast)

#####################################################################################################################################################################

def model_train():
    
    data = utils.get_data_from_excel('./FormattedData/formatted-data.xlsx')
    
    ''' Normalizing factors for pca'''
    for df in data:
        if(df != 'Stock returns'):
            data[df] = utils.normalize_dataframe(data[df])
    
            
    # Divide factors into training and testing datasets
    factor_training = {}
    factor_testing = {}
    for df in data:
        if(df != 'Stock returns'):
            factor_training[df], factor_testing[df] = utils.divide_df_lastyear(data[df])
    
    
    returns_training, returns_testing = utils.divide_df_lastyear(data['Stock returns'])
    returns_testing = np.exp(returns_testing) - 1 #Go back to simple percentage returns for better interpretability
    
    
    
# -------------------- PCA --------------------------------------------------------------------------------------------------
    
    #Dataframe for PCA on Macro and Fundamentals
    pca_factors_macro_fund = pd.concat([factor_training['Macro indices'], factor_training['Fundamentals']], axis=1)
    
    #Dataframe for PCA on Macro and Fundamentals for testing
    pca_factors_macro_fund_test = pd.concat([factor_testing['Macro indices'], factor_testing['Fundamentals']], axis=1)
    
    
    #PCA run over only macro and fund
    macro_fund_trans = pca_transform_wrapper(factors=pca_factors_macro_fund)
    
    #PCA run over only macro and fund for testingn
    macro_fund_trans_test = pca_transform_wrapper(factors=pca_factors_macro_fund_test)
    
    
    # Dataframe containig all the factors onto which to perform the last PCA run. 
    pca_factors_final = pd.concat([
                            factor_training['Rates returns'],
                            factor_training['Forex returns'],
                            factor_training['Commodities returns'],
                            macro_fund_trans], axis=1)
    
    # Dataframe containig all the factors on which to perform the last PCA run. (For testing) 
    pca_factors_final_test = pd.concat([
                            factor_testing['Rates returns'],
                            factor_testing['Forex returns'],
                            factor_testing['Commodities returns'],
                            macro_fund_trans_test], axis=1)
    
    
    #Final PCA-Transformed factors
    final_pca_trans= pca_transform_wrapper(factors=pca_factors_final)
    
    #Final PCA-Transformed factors for testing
    final_pca_trans_test = pca_transform_wrapper(factors=pca_factors_final_test)
    
    
    #Create lagged factors dataframe
    factors_lag = final_pca_trans.shift(periods=1, freq='B').dropna()
    factors_lag_test = final_pca_trans_test.shift(periods=1, freq='B').dropna()
    # Remove the first row of returns and the last row of factors (because of the shift)
    factors_lag = factors_lag.drop(factors_lag.index[-1])
    factors_lag_test = factors_lag_test.drop(factors_lag_test.index[-1])
    returns = returns_training.drop(returns_training.index[0])


#---------------------- MODEL TRAINING ---------------------------------------------------------------------------------

    X = factors_lag
    Y = returns
    
    residuals, weights, intercepts, forecast = linear_regression_model_train(Y, X, predict_data=factors_lag_test)

    #Make sure the matrix (S) contains only residuals variances, only diagonal 
    residuals_matrix = utils.force_diagonal_cov(residuals)

    
    
    
    # Covariance matrix of expected returns
    final_cov = np.transpose(weights.values)@factors_lag.cov().values@weights.values + residuals_matrix.values #(RESIDULAS PART IS WRONG!)
    
    
    
#-------------------------- OPTIMIZATION ---------------------------------------------------------------------------
    
    import optimize as op
    op_w = op.optimize_get_weights(final_cov)
    print("Optimized weights:")
    print(op_w)


    daily_portfolio_returns = returns_testing@(op_w)
    
    portfolio_daily_cumulative = daily_portfolio_returns.cumsum()
    
    portfolio_std_daily = np.sqrt(op_w.T @ final_cov @ op_w)

    portfolio_std_yearly = portfolio_std_daily * np.sqrt(daily_portfolio_returns.size)
    
    print(portfolio_daily_cumulative.tail())
    print("Portfolio volatility (std): ", portfolio_std_yearly) #This gives me the correct result, but why multiply by sqrt(number of days) ?
    print("Sharpe Ratio: ", portfolio_daily_cumulative.loc['2019-12-31'] / portfolio_std_yearly)
    
    #TODO I SHOULD PROBABLY CHECK THE CUMSUM OF EACH STOCK AND SEE IF IT MAKES SENSE TO HAVE THIS PORTFOLIO (OR JUST BUY THE FIRST STOCK)

    #IMPLEMENT OPTIMIZATION

    
    #TODO cross_val_score of all the linear models to see which one performs better
    #TODO calculate forecasted variance matrix and compare with the one calculated using weights
    #TODO once you have weekly data, perform optimization on the weekly data and check the portfolio
    #TODO for each week I should also calcualte the RMSE for the model.
    #TODO rebalance every week for the last year of data, for each week calculate what you need for the portfolio (return, variance, sharpe, var, svar)
    #     (returns and variance are calculated using the true data)
    
    #The idea is:
    # 1. calculate weights for current week, calculate portfolio weights and use them in that week.
    # 2. add that week to the training data and repeat
    # 3. do this for the last year kept as training data
    # 4. check how the portfolio performs in that week
    # 5. Then I can create the portfolio weight matrix for each week in that year (as excel file would be nice)
    
    #Maybe I can also do this in a monthly timeframe
    
        
    return
