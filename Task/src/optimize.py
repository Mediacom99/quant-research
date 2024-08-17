'''
This module is dedicated to the optimization of the risk-parity objective function.
'''

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from matplotlib import pyplot as plt

# Weights = portfolio weights

N_ASSETS = 5

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def risk_contribution(weights, cov_matrix):
    port_var = portfolio_variance(weights, cov_matrix)
    mrc = np.dot(cov_matrix, weights) #Marginal risk contribution
    rc = weights * mrc #Risk contribution
    return rc/port_var

def risk_parity_objective(weights, cov_matrix):
    rc = risk_contribution(weights, cov_matrix)
    pairwise_diffs = np.sum([(rc[i] - rc[j])**2 for i in range(N_ASSETS) for j in range(i+1, N_ASSETS)])
    return pairwise_diffs


def optimize_get_weights(cov_matrix):

    initial_weights = np.ones(N_ASSETS) / N_ASSETS #Start with equal weights 
    # Optimization constraints (positive weights and full invested)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(N_ASSETS))  # Weights between 0 and 1
    
    optimized_result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints, bounds=bounds)
    #print(f"Optimization results:\n{optimized_result}")
    
    return optimized_result.x




# Actual portfolio optimization (Every return is a log return)
def optimize_portfolio(cov_forecasted_returns: pd.DataFrame, returns_testing: pd.DataFrame):
    
    op_w = optimize_get_weights(cov_forecasted_returns)
    print("Optimized weights:")
    print(op_w)
    
    returns_testing_simple = np.exp(returns_testing) - 1
    
    # SIMPLE RETURNS COMBINE LINEARLY, LOG RETURNS DO NOT
    daily_portfolio_returns: pd.DataFrame = returns_testing_simple@(op_w)

    
    daily_portfolio_returns_log = np.log(daily_portfolio_returns + 1)
    
    print("Portfolio cumulative simple returns:", np.exp(daily_portfolio_returns_log.sum()) - 1)
        
    portfolio_var_from_daily = op_w.T @ cov_forecasted_returns @ op_w #Daily porfolio variance of the training dataset
    #Considering no covariance between trading days
    total_portfolio_var = portfolio_var_from_daily * daily_portfolio_returns.size #Portfolio variance by fixing weights over rolling window period
    
    return {
        'lreturn' : daily_portfolio_returns_log.sum(),
        'lvar' : total_portfolio_var,
    }
    

    
    #TODO I SHOULD PROBABLY CHECK THE CUMSUM OF EACH STOCK AND SEE IF IT MAKES SENSE TO HAVE THIS PORTFOLIO (OR JUST BUY THE FIRST STOCK)

    #IMPLEMENT OPTIMIZATION

    
    #TODO cross_val_score of all the linear models to see which one performs better
    #TODO calculate forecasted variance matrix and compare with the one calculated using weights
    #TODO once you have weekly data, perform optimization on the weekly data and check the portfolio
    #TODO rebalance every week for the last year of data, for each week calculate what you need for the portfolio (return, variance, sharpe, var, svar)
        # (returns and variance are calculated using the true data)
    #TODO I should give a weekly and monthly portfolio weight matrix. 
    #TODO Should also give results on different timeperiods based on the volatility of the variance of stock indices
    
    
    #The idea is:
    # 1. calculate weights for current week, calculate portfolio weights and use them in that week.
    # 2. add that week to the training data and repeat
    # 3. do this for the last year kept as training data
    # 4. check how the portfolio performs in that week
    # 5. Then I can create the portfolio weight matrix for each week in that year (as excel file would be nice)
    
    #Maybe I can also do this in a monthly timeframe
    