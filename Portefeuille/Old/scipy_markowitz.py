# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:19:15 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint


R0 = 0.05 # Risk-free rate

V0 = 0.0

# Expected returns and covariance matrix
mu = np.array([0.10, 0.12, 0.15, 0.08])  
sigma = np.array([
    [0.1, 0.02, 0.0, 0.03], 
    [0.02, 0.12, 0.01, -0.02], 
    [0.0, 0.01, 0.15, 0.03], 
    [0.03, -0.02, 0.03, 0.1]
])

get_var = lambda weights, sigma: weights.T.dot((sigma)).dot(weights)
get_return = lambda weights, mu, R0=0: weights.dot(mu)
get_var_rf = lambda weights, sigma: weights[1:].T.dot((sigma)).dot(weights[1:])
get_return_rf = lambda weights_rf, mu, R0: weights_rf[1:].dot(mu) + weights_rf[0]* R0

def mean_variance(weights, mu, sigma, c, R0=0):
    return c/2 * get_var(weights, sigma) - get_return(weights, mu)

def mean_variance_rf(weights_rf, mu, sigma, c, R0):
    return c/2 * get_var_rf(weights_rf, sigma) - get_return_rf(weights_rf, mu, R0)

def sharpe(weights, mu, sigma, R0 = 0):
    return get_return(weights, mu) / np.sqrt(get_var(weights,sigma))

def sharpe_rf(weights_rf, mu, sigma, R0):
    return (get_return_rf(weights_rf, mu, R0) - R0) / np.sqrt(get_var_rf(weights_rf,sigma))

def efficient_frontier(mu, sigma, R0, V0, num_points=100):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 8.0, num_points):  
        
        # Objective function 
        objective = lambda w: mean_variance(w, mu, sigma, c)

        init_weights = np.ones(n) / n

        # The sum of weights must be equal to 1
        constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)

        # Weights must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results


def efficient_frontier_rf(mu, sigma, R0, V0, num_points=100):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 4.0, num_points):  
        
        # Objective function 
        objective = lambda w: mean_variance_rf(w, mu, sigma, c, R0)

        init_weights = np.ones(n+1) / n+1

        # The sum of weights must be equal to 1
        constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

        # Weights must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n+1)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return_rf(res_weights, mu, R0)
            portfolio_risk_value = np.sqrt(get_var_rf(res_weights, sigma))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results

def optim_sharpe(mu, sigma, R0, V0):
    n = len(mu)
    # Objective function: minimize minus the Sharpe ratio
    objective = lambda w: -sharpe(w, mu, sigma)

    init_weights = np.ones(n) / n

    # The sum of weights must be equal to 1
    constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)

    # Weights must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(n)]

    result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

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

    # The sum of weights must be equal to 1
    constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

    # Weights must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(n+1)]

    result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

    std, ret = None, None
    if result.success:
        res_weights = result.x
        portfolio_return_value = get_return_rf(res_weights, mu, R0)
        portfolio_risk_value = np.sqrt(get_var_rf(res_weights, sigma))
        std, ret = portfolio_risk_value, portfolio_return_value

    return std, ret

ef_points = efficient_frontier(mu, sigma, R0, V0, 100)

tangent_std, tangent_ret = optim_sharpe(mu, sigma, R0, V0)

# Extracting risk and return values
stds = [p[0] for p in ef_points]
mus = [p[1] for p in ef_points]
sh = [mus[i]/stds[i] for i in range(len(stds))]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
#plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.scatter(stds, mus, c=sh, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.title('Markowitz Efficient Frontier without Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

ef_points_rf = efficient_frontier_rf(mu, sigma, R0, V0)

tangent_std_rf, tangent_ret_rf = optim_sharpe_rf(mu, sigma, R0, V0)

# Extracting risk and return values
stds_rf = [p[0] for p in ef_points_rf]
mus_rf = [p[1] for p in ef_points_rf]
sh_rf = [(mus_rf[i]-R0)/stds_rf[i] for i in range(len(stds_rf))]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
#plt.plot(stds_rf, mus_rf, label='Efficient Frontier', marker='o', linestyle='-')
plt.plot(tangent_std_rf, tangent_ret_rf, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.scatter(stds_rf, mus_rf, c=sh_rf, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.title('Markowitz Efficient Frontier with Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()