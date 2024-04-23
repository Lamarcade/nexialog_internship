# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:33:32 2024

@author: LoïcMARCADET
"""
import numpy as np
from scipy.optimize import minimize

class Portfolio:
    def __init__(self,mu,sigma, rf, short_sales = True):
        self.mu = mu
        self.sigma = sigma
        self.rf = rf
        self.n = len(mu)
        self.short_sales = short_sales
        
    def risk_free_stats(self):
        self.mu = np.insert(self.mu,0,self.rf)
        self.n = self.n+1
        sigma_rf = np.zeros((self.n, self.n))
        sigma_rf[1:,1:] = self.sigma
        self.sigma = sigma_rf
        
    def get_variance(self, weights):
        return weights.T.dot(self.sigma).dot(weights)
    
    def get_return(self, weights):
        return weights.T.dot(self.mu)
    
    def neg_mean_variance(self, weights, gamma):
        
        return 1/2 * self.get_variance(weights) - gamma * self.get_return(weights)
        
    def neg_sharpe(self, weights):
        return (- (self.get_return(weights) - self.rf)/ np.sqrt(self.get_variance(weights)))
    
    def weight_constraint(self):
        return({'type': 'eq', 'fun': lambda w: sum(w)-1})
    
    def bounds(self):
        if self.short_sales:
            bounds = [(-1.0, 1.0) for _ in range(self.n)]
        else:
            bounds = [(0.0, 1.0) for _ in range(self.n)]
        return bounds
    
    def init_weights(self):
        return(np.ones(self.n)/self.n)
    
    def optimal_portfolio(self,gamma):
        initial_weights = self.init_weights()
        constraint = self.weight_constraint()
        boundaries = self.bounds()
        result = minimize(self.neg_mean_variance, x0 = initial_weights, args = (gamma,), method = 'SLSQP', constraints = constraint, bounds = boundaries)
        if result.success:
            # Retrieve the optimal weights
            return(result.x)
    
    def efficient_frontier(self, n_points = 100, min_risk_tol = 0, max_risk_tol = 1):
        risks, returns, sharpes = [],[], []
        for gamma in np.linspace(min_risk_tol, max_risk_tol, n_points):
            weights = self.optimal_portfolio(gamma)
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