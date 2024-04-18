# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:03:09 2024

@author: LoïcMARCADET
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

get_var = lambda weights, sigma: weights.T.dot((sigma)).dot(weights)
get_return = lambda weights, mu, R0=0: weights.dot(mu)
get_var_rf = lambda weights, sigma: weights[1:].T.dot((sigma)).dot(weights[1:])
get_return_rf = lambda weights_rf, mu, R0: weights_rf[1:].dot(mu) + weights_rf[0]* R0

getsig = lambda w, sigma: np.sqrt(w@sigma@w)
getmu = lambda w, mu: w@mu

get_ESG = lambda weights, ESGs: weights.dot(ESGs) / sum(weights)
get_sharpe = lambda mus, stds, R0=0: [(mus[i] - R0)/stds[i] for i in range(len(stds))] 

def mean_variance(weights, mu, sigma, c, R0=0):
    return c/2 * get_var(weights, sigma) - get_return(weights, mu)

def mean_variance_rf(weights_rf, mu, sigma, c, R0):
    return c/2 * get_var_rf(weights_rf, sigma) - get_return_rf(weights_rf, mu, R0)

def mvESG(weights, mu, sigma, ESGs, c, ESG_pref):
    return c/2 * get_var(weights, sigma) - get_return(weights,mu) - ESG_pref(get_ESG(weights,ESGs))

def sharpe(weights, mu, sigma, R0 = 0):
    return (get_return(weights, mu) - R0) / np.sqrt(get_var(weights,sigma))

def sharpe_rf(weights_rf, mu, sigma, R0):
    return (get_return_rf(weights_rf, mu, R0) - R0) / np.sqrt(get_var_rf(weights_rf,sigma))


#%% Efficient frontiers with and without a risk-free asset
def efficient_frontier(mu, sigma, R0, V0, num_points=100):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 8.0, num_points):  
        
        # Objective function 
        objective = lambda w: mean_variance(w, mu, sigma, c)

        # Initial guess
        init_weights = np.ones(n) / n

        # The sum of weights must be equal to 1
        constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)

        # Weights must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            
            # Get the risk and return of the frontier portfolio
            portfolio_return_value = get_return(res_weights, mu)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results


def efficient_frontier_rf(mu, sigma, R0, V0, num_points=100):
    n = len(mu)
    results = []
    
    for c in np.linspace(0.0, 4.0, num_points):  
        
        objective = lambda w: mean_variance_rf(w, mu, sigma, c, R0)

        init_weights = np.ones(n+1) / n+1

        constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

        bounds = [(0.0, 1.0) for _ in range(n+1)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return_rf(res_weights, mu, R0)
            portfolio_risk_value = np.sqrt(get_var_rf(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results

#%% Find the optimal portfolio by maximizing the Sharpe ratio
def optim_sharpe(mu, sigma, R0, V0):
    n = len(mu)
    
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: -sharpe(w, mu, sigma)

    init_weights = np.ones(n) / n

    constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)

    bounds = [(0.0, 1.0) for _ in range(n)]

    result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

    # Return None for the risk and return in case the optimization fails
    std, ret = None, None
    
    if result.success:
        res_weights = result.x
        portfolio_return_value = get_return(res_weights, mu)
        portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
        std, ret = portfolio_risk_value, portfolio_return_value

    return std, ret

def optim_sharpe_rf(mu, sigma, R0, V0):
    n = len(mu)
    
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: 1/sharpe_rf(w, mu, sigma,R0)

    init_weights = np.ones(n+1) / (n+1)

    constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

    bounds = [(0.0, 1.0) for _ in range(n+1)]

    result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

    std, ret = None, None
    if result.success:
        res_weights = result.x
        portfolio_return_value = get_return_rf(res_weights, mu, R0)
        portfolio_risk_value = np.sqrt(get_var_rf(res_weights, sigma))
        std, ret = portfolio_risk_value, portfolio_return_value

    return std, ret

def optim_sharpe_esg(mu, sigma, ESGs, R0, V0, low_ESG, up_ESG, num_points = 100):
    n = len(mu)
    
    get_ESG_restr = lambda weights: weights.dot(ESGs) / sum(weights)
    
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: -sharpe(w, mu, sigma)

    init_weights = np.ones(n) / n

    constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)
    
    # Enforce an ESG score between low_ESG and up_ESG
    esg_constr = NonlinearConstraint(get_ESG_restr, lb = low_ESG, ub = up_ESG)

    bounds = [(0.0, 1.0) for _ in range(n)]

    result = minimize(objective, init_weights, method='SLSQP', constraints=[constraint, esg_constr], bounds=bounds)

    # Return None for the risk and return in case the optimization fails
    std, ret = None, None
    res_esg = []
    
    if result.success:
        res_weights = result.x
        portfolio_return_value = get_return(res_weights, mu)
        portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
        std, ret = portfolio_risk_value, portfolio_return_value
        res_esg.append(get_ESG(res_weights, ESGs))

    return std, ret, res_esg

#%% Efficient frontier with constraints on the ESG score
def target_esg_frontier(mu, sigma, ESGs, R0, V0, low_ESG, up_ESG, num_points = 100):
    n = len(mu)
    results = []
    
    # Function to use in the non-linear constraint
    # Note : It is linear by using a matrix of ESG weights
    get_ESG_restr = lambda weights: weights.dot(ESGs) / sum(weights)
    
    for c in np.linspace(0.0, 8.0, num_points):  
        
        objective = lambda w: mean_variance(w, mu, sigma, c)

        init_weights = np.ones(n) / n

        constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)
        
        # Enforce an ESG score between low_ESG and up_ESG
        esg_constr = NonlinearConstraint(get_ESG_restr, lb = low_ESG, ub = up_ESG)

        bounds = [(0.0, 1.0) for _ in range(n)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=[constraint, esg_constr], bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results

#%% Get the Sharpe ratio-ESG frontier
def esg_frontier(mu, sigma, ESGs, R0, V0, ESG_pref, num_points = 100):
    n = len(mu)
    results = []
    res_esg = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 8.0, num_points):  
        
        # Objective function is the negative mean-variance-ESG tradeoff
        objective = lambda w: mvESG(w, mu, sigma, ESGs, c, ESG_pref)
    
        init_weights = np.ones(n) / n
    
        constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)
    
        bounds = [(0.0, 1.0) for _ in range(n)]
    
        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)
    
        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))
            res_esg.append(get_ESG(res_weights, ESGs))
    
    return results, res_esg

