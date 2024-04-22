# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:19:25 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt

from portfolio_utils import *

# Expected returns and covariance matrix
mu = np.array([0.05, 0.07, 0.06, 0.10, 0.08])  
corr = np.array([
    [1.0, 0.7, 0.2, -0.3, 0.0], 
    [0.7, 1.0, 0.3, 0.2, 0.0], 
    [0.2, 0.3, 1.0, 0.1, 0.0],
    [-0.3, 0.2, 0.1, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1]
])
vol = [0.18, 0.2, 0.22, 0.25, 0.30]
sigma = corr.copy()
for i in range(5):
    for j in range(5):
        sigma[i][j] *= vol[i] * vol[j]

sigma_rf = rf_var(sigma)

rf = 0.03 # Risk-free rate
mu_rf = rf_mean(mu, rf)


#%%
ef_points = efficient_frontier(mu, sigma, 0, 100, short_sales = True)
stds, mus = [p[0] for p in ef_points], [p[1] for p in ef_points]
sh = get_sharpe(mus, stds, rf)

ef_p = efficient_frontier(mu_rf, sigma_rf, rf, 100, short_sales = True)
stds_rf, mus_rf = [p[0] for p in ef_p], [p[1] for p in ef_p]
sh_rf = get_sharpe(mus_rf, stds_rf, rf)

# =============================================================================
# # Draw random weights that sum to 1
# random_weights = np.random.rand(10000, 5)
# for i in range(10000):
#     random_weights[i,:] /= sum(random_weights[i,:])
# 
# random_sigma = [getsig(weights,sigma) for weights in random_weights]
# random_mu = [getmu(weights,mu) for weights in random_weights]
# =============================================================================

#%%
# Risk and return of the optimal portfolio
tangent_std, tangent_ret = optim_sharpe(mu, sigma, rf)

std_range = np.arange(0.001, 0.20, 0.01)
cml = capital_market_line(rf,tangent_ret, tangent_std, std_range)

#%% Plotting the efficient frontier

plt.figure(figsize=(8, 6))
#plt.scatter(stds_rf, mus_rf, c=sh_rf, cmap='viridis')
#plt.colorbar(label='Sharpe Ratio')
plt.plot(stds, mus)

plt.plot(std_range,cml, label = "CML", linestyle = "--")
#plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.title('Markowitz Efficient Frontier without a Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
#plt.scatter(stds_rf, mus_rf, c=sh_rf, cmap='viridis')
#plt.colorbar(label='Sharpe Ratio')
plt.plot(stds_rf, mus_rf)

plt.plot(std_range,cml, label = "CML", linestyle = "--")
#plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.title('Markowitz Efficient Frontier with a Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()
