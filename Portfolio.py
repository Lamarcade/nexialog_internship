# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:33:32 2024

@author: LoïcMARCADET
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import math
import seaborn as sns
import pandas as pd

class Portfolio:
    def __init__(self,mu,sigma, rf, short_sales = True, sectors = None):
        self.mu = mu
        self.sigma = sigma
        self.rf = rf
        self.n = len(mu)
        self.short_sales = short_sales
        self.sectors = sectors
        
        # Is the risk-free included in the mean and covariance
        self.rf_params = False
        
        self.existing_plot = False
        
    def risk_free_stats(self):
        if not(self.rf_params):
            self.mu = np.insert(self.mu,0,self.rf)
            self.n = self.n+1
            sigma_rf = np.zeros((self.n, self.n))
            sigma_rf[1:,1:] = self.sigma
            self.sigma = sigma_rf
            self.rf_params = True
            
            if self.sectors is not None:
                self.sectors.loc[-1] = ['RFA', 'RIS Risk-Free Asset']
                self.sectors.sort_index(inplace = True)
        
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
    
    def get_sharpe(self, weights):
        return ((self.get_return(weights) - self.rf)/ self.get_risk(weights))
    
    def set_sectors_composition(self,weights):
        
        if self.sectors is None:
            raise AttributeError('No sectors defined') 
            
        sectors_df = self.sectors.copy()
        
        dummy_weights = np.pad(weights, (0,len(sectors_df)-len(weights)))
        sectors_df['Weight'] = dummy_weights
        
        sector_weights = sectors_df.groupby('Sector')['Weight'].sum()
        
        # Total_weight should ideally be 1 with the optimization constraints
        total_weight = sector_weights.sum()
        sector_proportions = sector_weights / total_weight
        composition = pd.DataFrame({'Sector': sector_proportions.index, 'Weight': sector_proportions.values})
        composition['Acronym'] = composition['Sector'].str.extract(r'([A-Z]{3})')
        # Create dictionary with sector proportions
        self.sectors_composition = composition
    
    def neg_mean_variance(self, weights, gamma):
        # Negative mean-variance objective
        return 1/2 * self.get_variance(weights) - gamma * self.get_return(weights)
        
    def neg_sharpe(self, weights):
        return (- (self.get_return(weights) - self.rf)/ self.get_risk(weights))
    
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
            bounds = None
        elif input_bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.n)]
        else:
            bounds = input_bounds
        return bounds
    
    def init_weights(self):
        return(np.ones(self.n)/self.n)
    
    def optimal_portfolio(self, method = 1, gamma = None, max_risk = None, 
                          min_return = None, input_bounds = None,
                          new_constraints = None):
        initial_weights = self.init_weights()
        boundaries = self.bounds(input_bounds)
        constraints = [self.weight_constraint()]
        
        if new_constraints is not None:
            constraints.extend(new_constraints)
        
        if method == 3:
            # Minimum return constraint
            constraints.append(self.return_constraint(min_return))
            result = minimize(self.get_risk, x0 = initial_weights, 
                          method = 'SLSQP', 
                          constraints = constraints, bounds = boundaries)            
            
        elif method == 2:
            # Maximum volatility constraint
            constraints.append(self.risk_constraint(max_risk))
            result = minimize(self.neg_return, x0 = initial_weights, 
                          method = 'SLSQP', 
                          constraints = constraints, bounds = boundaries)            
            
        else:
            # Mean variance tradeoff
            result = minimize(self.neg_mean_variance, x0 = initial_weights, 
                          args = (gamma,), method = 'SLSQP', 
                          constraints = constraints, bounds = boundaries)
        return(result.x)
       
    
    def efficient_frontier(self, method = 1, n_points = 100, 
                           min_risk_tol = 0, max_risk_tol = 1,
                           min_std = 0.001, max_std = 0.2,
                           min_ret = 0.003, max_ret = 0.08,
                           new_constraints = None
                           ):
        risks, returns, sharpes = [],[], []

        if method == 3:
            for c in np.linspace(min_ret, max_ret, n_points):
                weights = self.optimal_portfolio(3, min_return = c, new_constraints = new_constraints)
                risk = self.get_risk(weights)
                if not(math.isnan(risk)):
                    risks.append(risk)
                    returns.append(self.get_return(weights))
                    sharpes.append(- self.neg_sharpe(weights))
        elif method == 2:
            for c in np.linspace(min_std, max_std, n_points): 
                weights = self.optimal_portfolio(2, max_risk= c, new_constraints = new_constraints)
                risk = self.get_risk(weights)

                if not(math.isnan(risk)):
                    risks.append(risk)
                    returns.append(self.get_return(weights))
                    sharpes.append(- self.neg_sharpe(weights))            
        else: 
            for gamma in np.linspace(min_risk_tol, max_risk_tol, n_points):
                weights = self.optimal_portfolio(1, gamma = gamma, new_constraints = new_constraints)
                risk = self.get_risk(weights)
                if not(math.isnan(risk)):
                    risks.append(risk)
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
    
    def new_figure(self, fig_size = (8,6), four_plots = False):
        sns.set_theme()
        if four_plots:
            self.fig, self.ax = plt.subplots((2,2), figsize=fig_size)
        else:
            self.fig, self.ax = plt.subplots(figsize=fig_size)
        self.existing_plot = False
        
    def make_title_save(self, figtitle, n_risky, savefig = False, score_source = None):
        if score_source is not None:
            title = score_source + figtitle + str(n_risky)
        else:
            title = figtitle + str(n_risky)
        if self.rf_params:
            title += '_risk_free'
        if not(self.short_sales):
            title += '_no_short'
    
        if savefig:
            plt.savefig('Figures/' + title + '.png')    
        
    def plot_frontier(self, rf_included, risks, returns, sharpes = None, marker_size = 1):
        if rf_included:
            if self.short_sales:
                self.ax.plot(risks, returns, linestyle = '--', label = 'CML')
            else:
                self.ax.plot(risks, returns, linestyle = '--', color = "g", label = 'CML no-short')
        else:
            ef = self.ax.scatter(risks, returns, c=sharpes, cmap='viridis', s = marker_size, label = 'Efficient frontier')
            self.fig.colorbar(ef, ax = self.ax, label='Sharpe Ratio')
        
        n_risky = len(self.mu)
        if rf_included:
            n_risky -= 1
        
        if not(self.existing_plot):
            self.ax.set_title('Markowitz Efficient Portfolios with ' + str(n_risky) + ' risky assets')
            self.ax.set_xlabel('Portfolio Risk')
            self.ax.set_ylabel('Portfolio Return')
            self.ax.grid(True)
            self.existing_plot = True
        self.ax.legend()
        
        figtitle = 'Efficient_frontiers_' + str(n_risky)
        plt.savefig('Figures/' + figtitle + '.png')
        
    def plot_tangent(self, tangent_risk, tangent_return):
        self.ax.plot(tangent_risk, tangent_return, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
        
        self.ax.legend()
        
    