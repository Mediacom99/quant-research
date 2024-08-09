'''
This module is dedicated to the optimization of the risk-parity objective function.
'''

import numpy as np
from scipy.optimize import minimize

# Weights = portfolio weights

n_assets = 5

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def risk_contribution(weights, cov_matrix):
    port_var = portfolio_variance(weights, cov_matrix)
    mrc = np.dot(cov_matrix, weights) #Marginal risk contribution
    rc = weights * mrc #Risk contribution
    return rc/port_var

def risk_parity_objective(weights, cov_matrix):
    rc = risk_contribution(weights, cov_matrix)
    pairwise_diffs = np.sum([(rc[i] - rc[j])**2 for i in range(n_assets) for j in range(i+1, n_assets)])
    return pairwise_diffs


def optimize_get_weights(cov_matrix):

    initial_weights = np.ones(n_assets) / n_assets #Start with equal weights 
    # Optimization constraints (positive weights and full invested)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(n_assets))  # Weights between 0 and 1
    
    optimized_result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints, bounds=bounds)
    return optimized_result.x

