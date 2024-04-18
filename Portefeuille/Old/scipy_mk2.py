# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:45:00 2024

@author: LoïcMARCADET
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint


R0 = 0.05 # Risk-free rate

# Expected returns and covariance matrix
mu = np.array([0.10, 0.12, 0.15, 0.08])  
sigma = np.array([
    [0.1, 0.02, 0.0, 0.03], 
    [0.02, 0.12, 0.01, -0.02], 
    [0.0, 0.01, 0.15, 0.03], 
    [0.03, -0.02, 0.03, 0.1]
])


def get_return(weights, mu, R0 = None, rf = False):
    if rf:
        return weights[1:].dot(mu) + weights[0]*R0
    else:
        return weights.dot(mu)
    
def get_var(weights, sigma, rf = False):
    if rf:
        return weights[1:].T.dot((sigma)).dot(weights[1:])
    else: 
        return weights.T.dot((sigma)).dot(weights)

def mean_variance(weights, sigma, c, R0 = None, rf = False):
    return c * get_var(weights, sigma,rf) - get_return(weights, mu, rf)

def efficient_frontier(mu, sigma, R0, rf = False, num_points=100):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 2.0, num_points):  
            
        # Objective function 
        objective = lambda w: mean_variance(w, sigma, c, R0, rf)

        if rf:
            init_weights = np.ones(n+1)/(n+1)
            # The sum of weights must be equal to 1
            constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

            # Weights must be between 0 and 1
            bounds = [(0.0, 1.0) for _ in range(n+1)]
        else:
            init_weights = np.ones(n) / n

            constraint = LinearConstraint(np.ones(n), lb=1.0, ub=1.0)

            bounds = [(0.0, 1.0) for _ in range(n)]


        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu, rf)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma, rf))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results

def efficient_frontier_rf(mu, sigma, R0, num_points=100):
    n = len(mu)
    results = []
    
    # Optimize the portfolio for different risk aversions
    for c in np.linspace(0.0, 2.0, num_points):  
            
        # Objective function 
        objective = lambda w: mean_variance(w, sigma, c, R0, rf = True)

        init_weights = np.ones(n+1)/(n+1)
        # The sum of weights must be equal to 1
        constraint = LinearConstraint(np.ones(n+1), lb=1.0, ub=1.0)

        # Weights must be between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n+1)]

        result = minimize(objective, init_weights, method='SLSQP', constraints=constraint, bounds=bounds)

        if result.success:
            res_weights = result.x
            portfolio_return_value = get_return(res_weights, mu, True)
            portfolio_risk_value = np.sqrt(get_var(res_weights, sigma, True))
            results.append((portfolio_risk_value, portfolio_return_value))

    return results

ef_points = efficient_frontier(mu, sigma, R0, False)

# Extracting risk and return values
stds = [p[0] for p in ef_points]
mus = [p[1] for p in ef_points]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('Markowitz Efficient Frontier without Risk-Free Asset')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

ef_points_rf = efficient_frontier_rf(mu, sigma, R0)
# Extracting risk and return values
stds_rf = [p[0] for p in ef_points]
mus_rf = [p[1] for p in ef_points]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('Markowitz Efficient Frontier with Risk-Free Asset')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()
