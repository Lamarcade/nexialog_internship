# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:33:32 2024

@author: LoïcMARCADET
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

class Portfolio:
    def __init__(self,mu,sigma, rf, short_sales = True):
        self.mu = mu
        self.sigma = sigma
        self.rf = rf
        self.n = len(mu)
        self.short_sales = short_sales
        
        # Is the risk-free included in the mean and covariance
        self.rf_params = False
        
        self.existing_plot = False
        
    def risk_free_stats(self):
        self.mu = np.insert(self.mu,0,self.rf)
        self.n = self.n+1
        sigma_rf = np.zeros((self.n, self.n))
        sigma_rf[1:,1:] = self.sigma
        self.sigma = sigma_rf
        self.rf_params = True
        
        return self
        
    def change_short_sales(self):
        self.short_sales = not(self.short_sales)
        return self
    
    def get_variance(self, weights):
        return weights.T.dot(self.sigma).dot(weights)
    
    def get_risk(self, weights):
        return np.sqrt(weights.T.dot(self.sigma).dot(weights))
    
    def get_return(self, weights):
        return weights.T.dot(self.mu)
    
    def neg_mean_variance(self, weights, gamma):
        
        return 1/2 * self.get_variance(weights) - gamma * self.get_return(weights)
        
    def neg_sharpe(self, weights):
        return (- (self.get_return(weights) - self.rf)/ np.sqrt(self.get_variance(weights)))

    
    def neg_return(self,weights):
        return - weights.T.dot(self.mu)
    
    def weight_constraint(self):
        return({'type': 'eq', 'fun': lambda w: sum(w)-1})
    
    def risk_constraint(self, max_risk):
        return({'type': 'ineq', 'fun': lambda w: max_risk - self.get_risk(w)})
    
    def return_constraint(self, min_return):
        return({'type': 'ineq', 'fun': lambda w: self.get_return(w) - min_return})
        #return(LinearConstraint(self.mu, lb = min_return, ub = np.inf))
    
    def bounds(self, input_bounds = None):
        if self.short_sales:
            bounds = [(-1.0, 1.0) for _ in range(self.n)]
        elif input_bounds == None:
            bounds = [(0.0, 1.0) for _ in range(self.n)]
        else:
            bounds = input_bounds
        return bounds
    
    def init_weights(self):
        return(np.ones(self.n)/self.n)
    
    def optimal_portfolio(self, method = 1, gamma = None, max_risk = None, min_return = None, input_bounds = None):
        initial_weights = self.init_weights()
        boundaries = self.bounds(input_bounds)
        constraint = self.weight_constraint()
        
        if method == 3:
            # Minimum return constraint
            constraint2 = self.return_constraint(min_return)
            result = minimize(self.get_risk, x0 = initial_weights, 
                          method = 'SLSQP', 
                          constraints = [constraint, constraint2], bounds = boundaries)            
            
        elif method == 2:
            # Maximum volatility constraint
            constraint2 = self.risk_constraint(max_risk)
            result = minimize(self.neg_return, x0 = initial_weights, 
                          method = 'SLSQP', 
                          constraints = [constraint, constraint2], bounds = boundaries)            
            
        else:
            result = minimize(self.neg_mean_variance, x0 = initial_weights, 
                          args = (gamma,), method = 'SLSQP', 
                          constraints = constraint, bounds = boundaries)
        return(result.x)
       
    
    def efficient_frontier(self, method = 1, n_points = 100, 
                           min_risk_tol = 0, max_risk_tol = 1,
                           min_std = 0.001, max_std = 0.2,
                           min_ret = 0.003, max_ret = 0.08
                           ):
        risks, returns, sharpes = [],[], []

        if method == 3:
            for c in np.linspace(min_ret, max_ret, n_points):
                weights = self.optimal_portfolio(3, min_return = c)
                risks.append(np.sqrt(self.get_variance(weights)))
                returns.append(self.get_return(weights))
                sharpes.append(- self.neg_sharpe(weights))
        elif method == 2:
            for c in np.linspace(min_std, max_std, n_points): 
                weights = self.optimal_portfolio(2, max_risk= c)
                risks.append(np.sqrt(self.get_variance(weights)))
                returns.append(self.get_return(weights))
                sharpes.append(- self.neg_sharpe(weights))            
        else: 
            for gamma in np.linspace(min_risk_tol, max_risk_tol, n_points):
                weights = self.optimal_portfolio(1, gamma)
                risks.append(np.sqrt(self.get_variance(weights)))
                returns.append(self.get_return(weights))
                sharpes.append(- self.neg_sharpe(weights))
        return risks, returns, sharpes
    
    def tangent_portfolio(self):
        initial_weights = self.init_weights()
        constraint = self.weight_constraint()
        boundaries = self.bounds()
        result = minimize(self.neg_sharpe, x0 = initial_weights, method = 'SLSQP', constraints = constraint, bounds = boundaries)
        if result.success:
            return(result.x)  
    
    def capital_market_line(self,risk_range):
        tangent_weights = self.tangent_portfolio()
        tangent_return = self.get_return(tangent_weights)
        tangent_risk = np.sqrt(self.get_variance(tangent_weights))
        return(self.rf + risk_range*(tangent_return - self.rf)/ tangent_risk)
    
    def random_weights(self, method = 'rand', dir_alpha = None, n_samples = 500):
        weights = np.zeros((n_samples, self.n))
        if method == 'Dirichlet' or method == 'dirichlet':
            if len(dir_alpha) != (self.n):
                raise ValueError("Invalid parameter length for the Dirichlet distribution")
            weights = np.random.dirichlet(dir_alpha, n_samples)
        
        else:
            weights = np.random.rand(n_samples, self.n)
            for i in range(n_samples):
                weights[i,:] /= sum(weights[i,:])
            
        random_sigma = [np.sqrt(self.get_variance(weight)) for weight in weights]
        random_mu = [self.get_return(weight) for weight in weights]
        return random_sigma, random_mu
    
    def new_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.existing_plot = False
        
    def plot_frontier(self, rf_included, risks, returns, sharpes = None, marker_size = 1):
        if rf_included:
            if self.short_sales:
                self.ax.plot(risks, returns, linestyle = '--', label = 'CML')
            else:
                self.ax.plot(risks, returns, linestyle = '--', color = "g", label = 'CML no-short')
        else:
            ef = self.ax.scatter(risks, returns, c=sharpes, cmap='viridis', s = marker_size, label = 'Efficient frontier')
            self.fig.colorbar(ef, ax = self.ax, label='Sharpe Ratio')
        
        if not(self.existing_plot):
            self.ax.set_title('Markowitz Efficient Portfolios')
            self.ax.set_xlabel('Portfolio Risk')
            self.ax.set_ylabel('Portfolio Return')
            self.ax.grid(True)
            self.existing_plot = True
        self.ax.legend()
        
    def plot_tangent(self, tangent_risk, tangent_return):
        self.ax.plot(tangent_risk, tangent_return, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
        
        self.ax.legend()
        
    
    