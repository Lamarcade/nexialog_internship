# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:03:09 2024

@author: LoïcMARCADET
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

#%% Add risk-free to the statistics

def rf_mean(mu, rf):
    return np.insert(mu,0,rf)

def rf_var(sigma):
    n = len(sigma)
    sigma_rf = np.zeros((n+1,n+1))
    sigma_rf[1:,1:] = sigma
    return sigma_rf

#%%
get_var = lambda weights, sigma: weights.T.dot((sigma)).dot(weights)
get_std = lambda weights, sigma: np.sqrt(weights.T.dot((sigma)).dot(weights))
get_return = lambda weights, mu, R0=0: weights.T.dot(mu)

getsig = lambda w, sigma: np.sqrt(w@sigma@w)
getmu = lambda w, mu: w@mu

#get_ESG = lambda weights, ESGs: weights.dot(ESGs) / sum(weights)

def get_ESG(weights, ESGs, rf):
    if rf > 0:
        if weights[0] < 0.99:
            return weights[1:].dot(ESGs) / sum(weights[1:])
        else:
            return 0
    else:
        return weights.dot(ESGs) / sum(weights)
    
get_sharpe = lambda mus, stds, R0: [(mus[i] - R0)/stds[i] for i in range(len(stds))] 


def mean_variance(weights, mu, sigma, c):
    # Opposite of the mean-variance objective, we want to minimize
    return c/2 * get_var(weights, sigma) - get_return(weights, mu)

def mean_variance_inv(weights, mu, sigma, gamma):
    # Put the risk tolerance instead of the risk aversion
    return 1/2 * get_var(weights, sigma) - gamma * get_return(weights, mu)

def mvESG(weights, mu, sigma, ESGs, rf, c, ESG_pref):
    return c/2 * get_var(weights, sigma) - get_return(weights,mu) - ESG_pref(get_ESG(weights,ESGs, rf))

def sharpe(weights, mu, sigma, R0):
    return (get_return(weights, mu) - R0) / np.sqrt(get_var(weights,sigma))

def sharpe2(weights, mu, sigma, R0):
    return (getmu(weights, mu) - R0) / getsig(weights,sigma)
#%% Efficient frontiers with and without a risk-free asset
def efficient_frontier(mu, sigma, R0, num_points=100, min_risk_tol = -0.5, max_risk_tol = 1):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(min_risk_tol, max_risk_tol, num_points):  
        
        # Objective function 
        objective = lambda w: mean_variance_inv(w, mu, sigma, c)

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

#%% Find the optimal portfolio by maximizing the Sharpe ratio
def optim_sharpe(mu, sigma, R0):
    n = len(mu)
    
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: -sharpe(w, mu, sigma, R0)

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

def optim_sharpe_esg(mu, sigma, ESGs, R0, low_ESG, up_ESG, num_points = 100):
    n = len(mu)
    
    get_ESG_restr = lambda weights: get_ESG(weights, ESGs, R0)
    
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: -sharpe(w, mu, sigma,R0)

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
        res_esg.append(get_ESG(res_weights, ESGs, R0))

    return std, ret, res_esg

def capital_market_line(rf, tangent_ret, tangent_std, std_range):
    return(rf + std_range * (tangent_ret - rf)/ tangent_std)

#%% Efficient frontier with constraints on the ESG score
def target_esg_frontier(mu, sigma, ESGs, R0, low_ESG, up_ESG, num_points = 100):
    n = len(mu)
    results = []
    # Function to use in the non-linear constraint
    # Note : It is linear by using a matrix of ESG weights
    get_ESG_restr = lambda weights: get_ESG(weights, ESGs, R0)
    
    for c in np.linspace(0.0, 8.0, num_points):  
        
        objective = lambda weights: mean_variance_inv(weights, mu, sigma, c)

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
def esg_frontier(mu, sigma, ESGs, R0, ESG_pref, num_points = 100):
    n = len(mu)
    results = []
    res_esg = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 8.0, num_points):  
        
        # Objective function is the negative mean-variance-ESG tradeoff
        objective = lambda w: mvESG(w, mu, sigma, ESGs, R0, c, ESG_pref)
    
        init_weights = np.ones(n) / n
    
        constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)
    
        bounds = [(0.0, 1.0) for _ in range(n)]
    
        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)
    
        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))
            res_esg.append(get_ESG(res_weights, ESGs, R0))
    
    return results, res_esg

#%% Random weights

def random_weights(n_assets, mu, sigma, method = 'rand', dir_alpha = None, n_samples = 500):
    weights = np.zeros((n_samples, n_assets +1))
    if method == 'Dirichlet' or method == 'dirichlet':
        if len(dir_alpha) != (n_assets +1):
            raise IndexError("Invalid parameter length for the Dirichlet distribution")
        weights = np.random.dirichlet(dir_alpha, n_samples)
    
    else:
        weights = np.random.rand(n_samples, n_assets + 1)
        for i in range(n_samples):
            weights[i,:] /= sum(weights[i,:])
        
    random_sigma = [np.sqrt(get_var(weight, sigma)) for weight in weights]
    random_mu = [get_return(weight, mu) for weight in weights]
    return random_sigma, random_mu
    