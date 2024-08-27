
"""

This module deals with training the model and forecasting returns based on factors
using PCA on factors and a linear multi-factor model.

"""


from matplotlib import pyplot as plt
import utils
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, SGDRegressor, ElasticNet, LassoLarsIC, TweedieRegressor, Ridge, TheilSenRegressor, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import logging as log

import optimize as op


logger = log.getLogger('model')

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
        logger.info("Loadings:\n%s", loadings)
    
    factor_df = pd.DataFrame(factor_transformed, columns=[f'PC{i}' for i in range(n_components)], index=factors.index)
    
    return factor_df    

# Actual PCA application for this particular case
def pca_run(factor_training: {pd.DataFrame}, print_loadings = False) -> pd.DataFrame:
    
    #Dataframe for PCA on Macro and Fundamentals
    pca_factors_macro_fund = pd.concat([factor_training['Macro indices'], factor_training['Fundamentals']], axis=1)
    
    #PCA run over only macro and fund
    macro_fund_trans = pca_transform_wrapper(factors=pca_factors_macro_fund)
    
    # Dataframe containig all the factors onto which to perform the last PCA run. 
    pca_factors_final = pd.concat([
                            factor_training['Rates returns'],
                            factor_training['Forex returns'],
                            factor_training['Commodities returns'],
                            macro_fund_trans], axis=1)

    #Final PCA-Transformed factors
    return pca_transform_wrapper(factors=pca_factors_final, print_loadings=print_loadings)    
    
    
    


#Linear regression of Y against X
# Y -> stock returns
# X -> pca-factors-lagged (1-d)
# SGDRegressor(loss='huber', shuffle=False)
def linear_regression_model_train(Y, X, model = LinearRegression(), x_for_predict = pd.DataFrame()):
     
    
    logger.info("current selected module is: %s", model)
    model = MultiOutputRegressor(model)
    
    logger.info("calculating weights")
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

    
   
    
    y_pred = model.predict(X)
    residuals = (Y - y_pred)
    #print("Residuals mean:")
    #print(residuals.mean())
    #print("Residuals std:") 
    #print(residuals.std())
    
    if (x_for_predict.empty):
        logger.info("no predict_data provided. Forecasted data will be returned empty.")
        return (residuals, coef_df, intercepts)
    else:
        forecast = model.predict(x_for_predict)
        forecast = pd.DataFrame(forecast, columns = Y.columns)
        return (residuals, coef_df, intercepts, forecast)


def cross_validation_regressors(Y: pd.DataFrame, X: pd.DataFrame):
    
    #models = {LinearRegression(), HuberRegressor(), SGDRegressor(loss='squared_epsilon_insensitive', shuffle=False, epsilon=0.05), ElasticNet(), Ridge(), TweedieRegressor(), PassiveAggressiveRegressor(), TheilSenRegressor()}
    
    models = [  
                LinearRegression(),
                PassiveAggressiveRegressor(),
                ElasticNet(),
                HuberRegressor(),
                TweedieRegressor(),
                SGDRegressor(shuffle=False),
                SGDRegressor(loss='squared_epsilon_insensitive', shuffle=False), 
                SGDRegressor(loss='squared_epsilon_insensitive', shuffle=False, penalty='elasticnet', epsilon=0.0006590374037441118),
    ]
    
    for model in models:
        multi_model = MultiOutputRegressor(model)  
        res = cross_val_score(multi_model, X = X,  y = Y,  cv = TimeSeriesSplit(n_splits=5), error_score='raise', scoring='r2')
        logger.info("score mean for %s is %s", model, res.mean())
        



def model_train(training_data: {pd.DataFrame}):
    '''
    Perform PCA and factors, run linear regression model, calculate expected returns covariance matrix and call the optimization module's functions
    to find the optimal portfolio weights
    
    Args:
        training_data: a collection of pandas dataframes containing factors and returns onto which perform PCA and run the regression mode
        
    Returns:
        The covariance matrix of forecasted returns
    '''
    
    factor_training = {}
    
    returns_training = training_data['Stock returns']
    
    # Normalize ONLY the factors for PCA
    for df in training_data:
        if(df != 'Stock returns'):
            factor_training[df] = utils.normalize_dataframe(training_data[df])
    
# -------------------- Principal Component Analysis --------------------------------------------------------------------------------------------------
    
    #Apply PCA on Macro and Fundamentals first, then with the rest
    final_pca_trans = pca_run(factor_training=factor_training, print_loadings=False)
    
# -------------------- Lag the factors --------------------------------------------------------------------------------------------------
    
    #Create lagged factors dataframe
    factors_lag = final_pca_trans.shift(periods=1, freq='B').dropna()
    # Remove the first row of returns and the last row of factors (because of the shift)
    factors_lag = factors_lag.drop(factors_lag.index[-1])
    returns = returns_training.drop(returns_training.index[0])


#---------------------- MODEL TRAINING ---------------------------------------------------------------------------------

    X = factors_lag
    Y = returns
    
    #print("Cross validation mean scores (higher is always better):\n")
    #Cross validation of regressors:
    #cross_validation_regressors(X,Y)
    
    # exposures are the result parameters of the fit (if residuals is smaller then epsilon then ignore it) if diff in daily is 0.001% then epsilon = is 1.001
    epsilon = 0.0001 * (Y.std().mean()) #Ignore differences smaller than 1/1000 of the average between the stds of the five stock indices.
    logger.info("Epsilon (loss function threshold): %s", epsilon)
    
    regression_model = SGDRegressor(loss='squared_epsilon_insensitive', shuffle=False, epsilon = epsilon)
    
    residuals, exposures, intercepts = linear_regression_model_train(Y, X, model = regression_model)
        
    #this matrix (S) contains only residuals variances, only diagonal 
    residuals_matrix = utils.force_diagonal_cov(residuals)
    
    # Covariance matrix of expected returns
    cov_forecasted_returns = exposures.T @ X.cov() @ exposures + residuals_matrix
    
    return cov_forecasted_returns

    

def run(OFFSET = pd.tseries.offsets.BYearEnd(1), divide_years = 16):
    '''
    This function is only called in run.py
    '''

    logger.info("MODEL.PY STARTING")

    #FIXME the relative path works only if the python script is run in the Task folder
    data = utils.get_data_from_excel("./FormattedData/formatted-data.xlsx")
    
    
    # Division betweenn training and datasets is 
    # done using a rolling window approach of 1 week
    
    returns = data['Stock returns']
    
    
    start_date  = returns.index.min()
    divide_date = start_date + pd.tseries.offsets.BYearEnd(divide_years)
    final_date  = returns.index.max()
    #final_date = divide_date + pd.tseries.offsets.BYearBegin(10)
    result = pd.DataFrame(columns=['Returns', 'Variance', 'Sharpe Ratio'])
    
    # The date I am using are probably wrong, I should check they work correctly
    ONEBDAY = pd.tseries.offsets.BDay(1)
    
    # ROLLING WINDOW OF 1 WEEK
    temp_date = divide_date
    while temp_date < final_date:
        
        logger.warning("offset start(test start) is %s", temp_date)
        logger.warning("testing end is %s", temp_date + OFFSET - ONEBDAY)
        logger.warning("training end is %s ", temp_date - ONEBDAY)
        
        #Get data from correct time frame
        returns_testing = returns.loc[temp_date:temp_date + OFFSET - ONEBDAY]
        training_data = utils.offset_dataframe_collection(data, start_date = start_date, end_date = temp_date)
        
        #Calculate covariance matrix of expected returns
        cov_matrix_expected_returns = model_train(training_data=training_data)
        
        logger.critical("actual number of days: %s", returns_testing['Indice Azionario Paese 1'].size)
        
        #Optimize the portfolio and check performance against testing dataset
        res = op.optimize_portfolio(cov_matrix_expected_returns, returns_testing)
        
        returns_testing_simple_max = np.abs((np.exp(returns_testing.sum()) - 1)).max()
        logger.warning("singular stock biggest simple total returns (SHOULD ADD TOTAL VARIANCE): %s\n\n", returns_testing_simple_max)
        
        if(returns_testing_simple_max < np.abs((np.exp(res['lreturn']) - 1)) ):
            logger.critical("PORTFOLIO RETURN IS BIGGER THAN BIGGEST SINGLE STOCK!!!")
            exit(1)
        
        #Append result to result dataframe
        result.loc[temp_date] = [res['lreturn'], res['lvar'], 0]
        
        #Go to next period
        temp_date += OFFSET
        
    
    
    
    #Dataframe with portfolio returns and variance for each rolling period
    portfolio_log_returns = result['Returns']
    portfolio_log_var = result['Variance']
    
    portfolio_return_simple_tot = np.exp(portfolio_log_returns.sum()) - 1
    
    df_portfolio_simple_returns = np.exp(portfolio_log_returns.cumsum()) - 1
    
    
    #FIXME Plot stock cum returns against portfolio cum returns (Works only with daily upgrading)
    returns_testing_simple: pd.DataFrame = np.exp(returns.loc[divide_date:final_date].cumsum()) - 1
    df = pd.concat([df_portfolio_simple_returns, returns_testing_simple], axis=1)
    print(df.tail())
    series_to_highlight = 'Returns'
    for column in df.columns:
        if column != series_to_highlight:
            plt.plot(df.index, df[column], label=column, alpha=0.5)
    # Highlight specific series
    plt.plot(df.index, df[series_to_highlight], linewidth=2, color='blue', label=f"{series_to_highlight} (highlighted)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    #Standard deviation of the portfolio over the total testing period (error propagated from log returns)
    portfolio_std_simple_tot = np.sqrt(portfolio_log_var.sum())
    
    print("Total portfolio return over testing period: ", portfolio_return_simple_tot)
    print("Total portfolio volatility over testing period: ", portfolio_std_simple_tot)
    print("Sharpe Ratio over testing period: ", portfolio_return_simple_tot / portfolio_std_simple_tot)
    print("MAx single stock return over whole testing period:\n",(np.exp(returns.loc[divide_date:final_date].sum()) - 1).max())
