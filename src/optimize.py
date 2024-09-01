'''
This module is dedicated to the optimization of the portfolio
using a risk-parity approach where each asset's contribution
to the total portfolio volatility is the same.
'''

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import logging as log

logger = log.getLogger('optimize')

N_ASSETS = 5

def portfolioVariance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def riskContribution(weights, cov_matrix):
    port_var = portfolioVariance(weights, cov_matrix)
    mrc = np.dot(cov_matrix, weights) #Marginal risk contribution
    rc = weights * mrc #Risk contribution
    return rc/port_var

def riskParityObjective(weights, cov_matrix):
    rc = riskContribution(weights, cov_matrix)
    pairwise_diffs = np.sum([(rc[i] - rc[j])**2 for i in range(N_ASSETS) for j in range(i+1, N_ASSETS)])
    return pairwise_diffs


def optimizationRun(cov_matrix):

    #Start with equal weights
    initial_weights = np.ones(N_ASSETS) / N_ASSETS

    # Optimization constraints (positive weights and full invested)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(N_ASSETS))  # Weights between 0 and 1

    optimized_result = minimize(riskParityObjective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints, bounds=bounds)

    return optimized_result.x



# Actual portfolio optimization (Every return is a log return)
def optimizePortfolioRun(cov_expected_returns: pd.DataFrame, returns_testing: pd.DataFrame):
    """
    Actual portfolio optimization specific implementation. Calculates
    the total portfolio log return and total portfolio variance in the chosen rolling window timeframe.

    Args:
    cov_expected_returns: covariance matrix of expected returns (product of multifactor model)
    returns_testing: dataframe of actual returns to use with the optimized portfolio weights

    Returns:
    structure containing:
    lreturn: total portfolio log return
    lvar: total portfolio var
    """

    op_w = optimizationRun(cov_expected_returns)
    logger.info("Optimized weights:")
    logger.info(op_w)
    print(f"{returns_testing.index.min().date()} --> {returns_testing.index.max().date()}", end="\t")
    print(*[f"{w:.2%}" for w in op_w], sep=" | ")

    returns_testing_simple = np.exp(returns_testing) - 1

    # SIMPLE RETURNS COMBINE LINEARLY, LOG RETURNS DO NOT
    tf_portfolio_returns = returns_testing_simple@(op_w)

    # Rolling window timeframe (tf) portfolio log returns
    tf_portfolio_returns_log = np.log(tf_portfolio_returns + 1)

    logger.critical("portfolio cumulative simple returns: %s", np.exp(tf_portfolio_returns_log.sum()) - 1)

    #Daily porfolio variance of the training dataset
    portfolio_var_from_tf = op_w.T @ cov_expected_returns @ op_w
    
    #Considering no covariance between trading days
    #Portfolio variance by fixing weights over rolling window period
    total_portfolio_var_over_tf = portfolio_var_from_tf * tf_portfolio_returns.size

    return {
        'sreturn' : tf_portfolio_returns,
        'lvar' : total_portfolio_var_over_tf,
        'weights' : op_w 
    }
