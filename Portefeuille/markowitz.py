# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:19:25 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt

from portfolio_utils import *

R0 = 0.02 # Risk-free rate

# Expected returns and covariance matrix
mu = np.array([0.10, 0.12, 0.15, 0.08])  
sigma = np.array([
    [0.1, 0.02, 0.0, 0.03], 
    [0.02, 0.12, 0.01, -0.02], 
    [0.0, 0.01, 0.15, 0.03], 
    [0.03, -0.02, 0.03, 0.1]
])
sigma_rf = rf_var(sigma)

rf = 0.02
mu_rf = rf_mean(mu, rf)

ef_points = efficient_frontier(mu, sigma, 0, 100)
stds, mus = [p[0] for p in ef_points], [p[1] for p in ef_points]
sh = get_sharpe(mus, stds, rf)

ef_p = efficient_frontier(mu_rf, sigma_rf, rf, 100)
stds_rf, mus_rf = [p[0] for p in ef_p], [p[1] for p in ef_p]
sh_rf = get_sharpe(mus_rf, stds_rf, rf)

# Draw random weights that sum to 1
random_weights = np.random.rand(10000, 5)
for i in range(10000):
    random_weights[i,:] /= sum(random_weights[i,:])

random_sigma = [getsig(weights,sigma_rf) for weights in random_weights]
random_mu = [getmu(weights,mu_rf) for weights in random_weights]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
plt.scatter(stds, mus, c=sh, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.plot(stds_rf,mus_rf, label = "with a Risk-free asset")
plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
plt.title('Markowitz Efficient Frontier with and without a Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()
