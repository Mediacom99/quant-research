
"""
This module deals with training the model, calculating the covariance matrix
of expected returns and optimizing the portfolio based on this matrix. The model uses PCA on different factors and a linear multifactor model to explain the relation between factors and stock indices returns.
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

N_PCA_COMPONENTS = 5

def pcaTransformWrap(factors: pd.DataFrame, print_loadings: bool, cols_name_add: str) -> pd.DataFrame:
    """
    Calculates Principal Component Analysis over given dataframe, choosing the first N_PCA_COMPONENTS
    if print_loadings is True it also prints the explained variance of each PC and a bar plot
    of the factor loadings.

    Args:
    factors : DataFrame containing the data onto which to perform PCA
    N_PCA_COMPONENTS: number of principal components to keep (columns of returned dataframe, rows is num of samples)
    print_loadings : wether or not to print factor loadings and do a bar plot

    Returns:
    dataframe containig the calculated principal components
    """

    factors_names = factors.columns
    pca = PCA(N_PCA_COMPONENTS)
    factor_transformed = pca.fit_transform(factors) #New factor_training dataset after PCA

    cols = [f'{cols_name_add} PC{i}' for i in range(N_PCA_COMPONENTS)]

    # loadings are the principal axes in the original factors space
    loadings = pd.DataFrame(
    pca.components_.T,
    columns=cols,
    index=factors_names
    )

    if(print_loadings):
        print("Factor loadings:");
        print("Explained variance percentage:", pca.explained_variance_ratio_)
        print(f"Total explained variance percentage: {(pca.explained_variance_ratio_.sum()*100):.2f}%")
        loadings.plot(kind='bar')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylabel('Contribution (1 = 100%)', fontsize=16)
        plt.title('Factors loadings contribution to principal components', fontsize=18)
        plt.show()

    factor_df = pd.DataFrame(factor_transformed, columns=cols, index=factors.index)

    return factor_df


def pcaSpecificRun(factor_training: {pd.DataFrame}, print_loadings: bool) -> pd.DataFrame:
    """
    Special application of pca_transform_wrapper for this particular dataset.
    It applies PCA first between macroeconomic indices and fundamentals. Then between
    that result and the rest of the data. This last result's principal components
    are returned.

    Args:
    factor_training : portion of factors data chosen for training purposes (collection of dataframes)
    print_loadings : wether or not to print factor loadings (see function pca_transform_wrapper)

    Returns: PCA-transformed factors in a DataFrame
    """

    #Dataframe for PCA on Macro and Fundamentals
    pca_factors_macro_fund = pd.concat([factor_training['Macro indices'], factor_training['Fundamentals']], axis=1)

    #PCA run over macro and fund only
    macro_fund_trans = pcaTransformWrap(factors=pca_factors_macro_fund, print_loadings=print_loadings, cols_name_add = 'Macro-Fund')

    # Dataframe containig all the factors onto which to perform the last PCA run.
    pca_factors_final = pd.concat([
                            factor_training['Rates returns'],
                            factor_training['Forex returns'],
                            factor_training['Commodities returns'],
                            macro_fund_trans], axis=1)

    #Final PCA-Transformed factors
    final_pca_transformed = pcaTransformWrap(factors=pca_factors_final, print_loadings=print_loadings, cols_name_add = 'Final')
    return final_pca_transformed




def crossValidationRegressors(Y: pd.DataFrame, X: pd.DataFrame):
    """
    Perform K-Fold Cross Validation using a TimeSeriesSPlit.
    It divides data keeping track of temporal relations and performs
    cross-validation of different models using for each their
    own preferred scoring method. Each scoring method is
    standardized so that higher value means better score.

    Args:
    Y, X data onto which to perform linear regression
    """

    models = [
                LinearRegression(),
                PassiveAggressiveRegressor(),
                ElasticNet(),
                HuberRegressor(),
                TweedieRegressor(),
                SGDRegressor(loss='huber', shuffle=False),
                SGDRegressor(loss='epsilon_insensitive', shuffle=False),
                SGDRegressor(loss='squared_epsilon_insensitive', shuffle=False, penalty='elasticnet', epsilon = 0.0001 * Y.std().mean() ),
    ]

    for model in models:
        multi_model = MultiOutputRegressor(model)
        res = cross_val_score(multi_model, X = X,  y = Y,  cv = TimeSeriesSplit(n_splits=5), error_score='raise', scoring='r2')
        print(f"score mean for {model} is {res.mean()}")






def regressionModelRun(Y, X, model = LinearRegression(), x_for_predict = pd.DataFrame()):
    """
    Performs linear regression using either the default LinearRegression regressor or any valid regressor
    models passed as input.

    Args:
    Y, X: datasets for regression of Y=Bx + C
    model: regression model to use
    x_for_predict: testing data to give the trained model to forecast some data

    Returns:
    (residuals, coef_df, intercepts, forecast) : residuals, calculated weights, calculated intercepts, forecasted data
    """


    logger.info("current selected module is: %s", model)
    model = MultiOutputRegressor(model)

    logger.info("calculating weights")
    model.fit(X, Y)

    # Get coefficients and intercepts
    coefficients = np.array([estimator.coef_ for estimator in model.estimators_])
    intercepts = np.array([estimator.intercept_ for estimator in model.estimators_])


    # Create a DataFrame of coefficients (factors exposures)
    coef_df = pd.DataFrame(coefficients.T,
                        columns=Y.columns,
                        index=X.columns)

    intercepts = pd.DataFrame(intercepts, columns=['Intercepts'], index = Y.columns)

    y_pred = model.predict(X)
    residuals = (Y - y_pred)

    #TODO HERE I CAN DO THINGS WITH RESIDUALS
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



def getCovMatrixFutureReturns(training_data: {pd.DataFrame}, print_pca_factor_loadings: bool, do_cross_validation: bool):
    '''
    Runs the following steps tp calculate the covariance matrix of future returns based on the weights and residuals
    of the linear regression.

    Steps:
    1. Perform PCA on the factors
    2. Lag factors dataframe
    3. perform linear regression, get factor exposures(coefficients)
       and residuals.
    4. return covariance matrix of future returns

    Args:
        training_data: a collection of pandas dataframes containing factors and returns onto which perform PCA and run the regression mode
        print_pca_factor_loadings: wether or not to print PCA bar graphs and explained variance info
        do_cross_validation: wether or not to cross validate a list of linear models.

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
    final_pca_trans = pcaSpecificRun(
        factor_training=factor_training,
        print_loadings = print_pca_factor_loadings
    )

# -------------------- Lag the factors --------------------------------------------------------------------------------------------------

    #Create lagged factors dataframe by one day in the future (data of day before used to calculate returns for current day considered)
    factors_lag = final_pca_trans.shift(periods=1, freq='B').dropna()

    # Aligning factors and returns by remove the first row of returns and the last row of factors
    factors_lag = factors_lag.drop(factors_lag.index[-1])
    returns = returns_training.drop(returns_training.index[0])

#---------------------- MODEL TRAINING ---------------------------------------------------------------------------------

    X = factors_lag
    Y = returns

    if(do_cross_validation):
        print("Cross validation mean scores (higher is always better):")
        crossValidationRegressors(X,Y)

    # exposures are the result parameters of the fit (if residuals is smaller then epsilon then ignore it) if diff in daily is 0.001% then epsilon = is 1.001
    epsilon = 0.0001 * (Y.std().mean()) #Ignore differences smaller than 1/1000 of the average between the stds of the five stock indices.
    logger.info("Epsilon (loss function threshold): %s", epsilon)

    regression_model = SGDRegressor(
        loss='squared_epsilon_insensitive',
        penalty = 'elasticnet',
        shuffle=False,
        epsilon = epsilon
    )

    residuals, exposures, intercepts = regressionModelRun(Y, X, model = regression_model)

    #this matrix (S) contains only residuals variances, only diagonal
    residuals_matrix = utils.forceDiagonalCov(residuals)

    # Covariance matrix of expected returns
    cov_future_returns = exposures.T @ X.cov() @ exposures + residuals_matrix

    return cov_future_returns




def tradingModelRun(formattedDataPath: str, OFFSET: pd.tseries.offsets, print_pca_factor_loadings: bool, do_cross_validation: bool, divide_years = 16):
    '''
    This function runs the trading model with a rolling window approach defined by the OFFSET argument.
    The data is initially divided keeping divide_years of business years as training datta, the first testing
    data will be whatever the OFFSET is. For each run, the training data is updated with the testing data
    just used, the testing data is the next OFFSET.

    Args:
    formattedDataPath : path to the previously generated formatted-data.xlsx file
    OFFSET: tseries.offset, rolling window period
    print_pca_factor_loadings: wether  to print PCA graphs and explained variance
    do_cross_validation: wether to perform cross validation of different regressors

    Returns:
    print-only function.
    '''

    logger.info("MODEL.PY STARTING")

    #Load data from provided path
    data = utils.get_data_from_excel(formattedDataPath)

    returns = data['Stock returns']

    start_date  = returns.index.min()
    divide_date = start_date + pd.tseries.offsets.BYearEnd(divide_years)
    final_date  = returns.index.max()
    ONEBDAY = pd.tseries.offsets.BDay(1)
    temp_date = divide_date

    trading_model_result = pd.DataFrame(
        columns=['Portfolio Returns', 'Portfolio Variance', 'Sharpe Ratio']
    )
    
    portfolio_matrix = pd.DataFrame(
        columns = returns.columns,
    )

    # Rolling window loop
    while temp_date < final_date:
        #Filter data given rolling window timeframe
        returns_testing = returns.loc[temp_date:temp_date + OFFSET - ONEBDAY]
        training_data = utils.timeFilterDataframeCollection(
            data,
            start_date = start_date,
            end_date = temp_date
        )

        #Calculate covariance matrix of expected returns
        cov_matrix_expected_returns = getCovMatrixFutureReturns(
            training_data=training_data,
            print_pca_factor_loadings=print_pca_factor_loadings,
            do_cross_validation=do_cross_validation
        )

        #Optimize the portfolio and check performance against testing dataset
        optimize_result = op.optimizePortfolioRun(cov_matrix_expected_returns, returns_testing)

        #Append result to final result dataframe
        trading_model_result.loc[temp_date] = [optimize_result['lreturn'], optimize_result['lvar'], 0]

        #Append weights to portfolio matrix
        portfolio_matrix.loc[temp_date] = optimize_result['weights']
        
        logger.warning("offset start(test start) is %s", temp_date)
        logger.warning("testing end is %s", temp_date + OFFSET - ONEBDAY)
        logger.warning("training end is %s", temp_date - ONEBDAY)
        logger.warning("Number of testing days (size of ): %s\n",
                       returns_testing['Indice Azionario Paese 1'].size)
        _returns_testing_simple_max = np.abs((np.exp(returns_testing.sum()) - 1)).max()
        logger.warning("singular stock biggest total cumulative return: %s\n\n",
                       _returns_testing_simple_max)

        #Roll window
        temp_date += OFFSET


    portfolio_log_cum_returns = trading_model_result['Portfolio Returns']
    portfolio_variance = trading_model_result['Portfolio Variance']

    portfolio_simple_cum_return_total = np.exp(portfolio_log_cum_returns.sum()) - 1
    portfolio_simple_cum_returns = np.exp(portfolio_log_cum_returns.cumsum()) - 1

    returns_testing_cum_simple: pd.DataFrame = np.exp(returns.loc[divide_date:final_date].cumsum()) - 1

    #Standard deviation of the portfolio over the total testing period
    portfolio_volatility = np.sqrt(portfolio_variance.sum())

    print(f"Total portfolio return over testing period: {(portfolio_simple_cum_return_total*100):.2f}%")
    print(f"Total portfolio volatility over testing period: {(portfolio_volatility*100):.2f}%")
    print(f"Sharpe Ratio over testing period: {portfolio_simple_cum_return_total / portfolio_volatility:.2f}")
    print(f"Max single cumulative stock return over whole testing period: {((np.exp(returns.loc[divide_date:final_date].sum()) - 1).max()*100):.2f}%")
    print(f"Min single cumulative stock return over whole testing period: {((np.exp(returns.loc[divide_date:final_date].sum()) - 1).min()*100):.2f}%")
    #TODO print max and min variance

    #Here the two inputs might be in different timeframes (daily, monthly)
    # it depends on the chosen rebalancing frequency
    utils.graphPortfolioStocksPerformance(
        portfolio_simple_cum_returns,
        returns_testing_cum_simple
    )

    portfolio_matrix.to_excel("../portfolio-matrices/portfolio-matrixBDay1.xlsx")
    utils.graphPortfolioWeights(portfolio_matrix)
    return
