# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:57:23 2024

@author: LoïcMARCADET
"""
import numpy as np
import matplotlib.pyplot as plt

from portfolio_utils import *

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

# ESG scores of the assets
ESGs = np.array([0.90, 0.70, 0.20, 0.15]) 

# ESG preference functions
f = lambda s: s/10
g = lambda s: s+1


#%% Plots

# Standard efficient frontier without risk-free asset
ef_points = efficient_frontier(mu, sigma, R0, V0, 100)
stds, mus = [p[0] for p in ef_points], [p[1] for p in ef_points]
sh = get_sharpe(mus, stds, R0)

# Risk and return of the optimal portfolio
tangent_std, tangent_ret = optim_sharpe(mu, sigma, R0, V0)

# Standard efficient frontier with risk-free asset
ef_points_rf = efficient_frontier_rf(mu, sigma, R0, V0, 100)
stds_rf, mus_rf = [p[0] for p in ef_points_rf], [p[1] for p in ef_points_rf]
sh_rf = get_sharpe(mus_rf, stds_rf, R0)


# Efficient frontier with target ESG scores
low_ESG = 0.7
up_ESG = 1
aim_points = target_esg_frontier(mu, sigma, ESGs, R0, V0, low_ESG, up_ESG)

stdsa, musa = [p[0] for p in aim_points], [p[1] for p in aim_points]
sha = get_sharpe(musa, stdsa)

# Sharpe ratio - ESG frontier
esg_points, res_esg = esg_frontier(mu, sigma, ESGs, R0, V0, f)

stdse, muse = [p[0] for p in esg_points], [p[1] for p in esg_points]
she = get_sharpe(muse, stdse)

# Draw random weights that sum to 1
random_weights = np.random.rand(10000, 4)
for i in range(10000):
    random_weights[i,:] /= sum(random_weights[i,:])

random_sigma = [getsig(weights,sigma) for weights in random_weights]
random_mu = [getmu(weights,mu) for weights in random_weights]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
#plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.scatter(stds, mus, c=sh, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
plt.plot(stds_rf, mus_rf)
plt.title('Markowitz Efficient Frontier without and with Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(stds, mus, c=sh, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
#plt.scatter(stdse, muse, c=she)
plt.plot(stdsa, musa, label='Efficient Frontier with ' + str(low_ESG) + ' <= ESG <= ' + str(up_ESG), marker='o', linestyle='-')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

# Plotting the ESG efficient frontier
plt.figure(figsize=(8, 6))
plt.plot(res_esg, she, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('ESG Efficient Frontier without Risk-Free Asset')
plt.xlabel('ESG score')
plt.ylabel('Sharpe ratio')
plt.grid(True)
plt.legend()
plt.show()

#%% Maximum Sharpe for varying constraints
stds_c, mus_c, sh_esgs = [],[],[]
for lESG in np.arange(0,0.9,0.05):
    stp, mup, sesg = optim_sharpe_esg(mu, sigma, ESGs, R0, V0, lESG, 1, num_points = 100)
    stds_c.append(stp)
    mus_c.append(mup)
    sh_esgs.append(sesg)
    
sh_c = get_sharpe(mus_c, stds_c, R0)    
    
plt.figure(figsize=(8, 6))
plt.plot(sh_esgs, sh_c, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('ESG Constraints impact on Sharpe Ratio')
plt.xlabel('ESG score')
plt.ylabel('Sharpe ratio')
plt.grid(True)
plt.legend()
plt.show()


