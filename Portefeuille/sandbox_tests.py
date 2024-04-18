# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:11:28 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy.random
import pandas as pd

from portfolio_utils import *

stocks = pd.read_csv("sp500_stocks.csv")

# Convert 'Date' column to datetime
stocks['Date'] = pd.to_datetime(stocks['Date'])

# Pivot the DataFrame to have symbols as columns
stocks_pivot = stocks.pivot(index='Date', columns='Symbol', values='Adj Close')

# Calculate monthly returns for each symbol
monthly_returns = stocks_pivot.resample('ME').ffill().pct_change()

monthly_returns = monthly_returns.iloc[1:]

# Drop columns that are at least 50 NaN
monthly_returns = monthly_returns.dropna(axis=1, thresh=50)

#%%
n_assets = 15
mr = monthly_returns.iloc[:, :n_assets]

means = np.array(mr.mean(axis = 0))
cov = np.array(mr.cov())

#%%
rf = 0.002 # Risk-free rate

means_rf = rf_mean(means, rf)
cov_rf = rf_var(cov)
# ESG scores of the assets
ESGs = np.random.random(n_assets)

# ESG preference functions
f = lambda s: s/10
g = lambda s: s+1

#%%
# Standard efficient frontier without risk-free asset
ef_points = efficient_frontier(means_rf, cov_rf, rf, 100)
stds, mus = [p[0] for p in ef_points], [p[1] for p in ef_points]
sh = get_sharpe(mus, stds, rf)

# Risk and return of the optimal portfolio
tangent_std, tangent_ret = optim_sharpe(means_rf, cov_rf, rf)

# Efficient frontier with target ESG scores
low_ESG = 0.5
up_ESG = 1
aim_points = target_esg_frontier(means_rf, cov_rf, ESGs, rf, low_ESG, up_ESG)

stdsa, musa = [p[0] for p in aim_points], [p[1] for p in aim_points]
sha = get_sharpe(musa, stdsa, rf)

# Draw random weights that sum to 1
random_weights = np.random.rand(5000, n_assets + 1)
for i in range(5000):
    random_weights[i,:] /= sum(random_weights[i,:])

random_sigma = [np.sqrt(get_var(weights,cov_rf)) for weights in random_weights]
random_mu = [get_return(weights,means_rf) for weights in random_weights]

# Plotting the efficient frontier
plt.figure(figsize=(8, 6))
#plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.scatter(stds, mus, c=sh, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')

plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
#plt.scatter(stdse, muse, c=she)
#plt.plot(stdsa, musa, label='Efficient Frontier with ' + str(low_ESG) + ' <= ESG <= ' + str(up_ESG), marker='o', linestyle='-')
plt.title('Markowitz Efficient Frontier without Risk-Free Asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

#%%
stds_c, mus_c, sh_esgs = [],[],[]
for lESG in np.arange(0,0.9,0.05):
    stp, mup, sesg = optim_sharpe_esg(means_rf, cov_rf, ESGs, rf, lESG, 1, num_points = 100)
    stds_c.append(stp)
    mus_c.append(mup)
    sh_esgs.append(sesg)
    
sh_c = get_sharpe(mus_c, stds_c, rf)    
    
plt.figure(figsize=(8, 6))
plt.plot(sh_esgs, sh_c, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('ESG Constraints impact on Sharpe Ratio')
plt.xlabel('ESG score')
plt.ylabel('Sharpe ratio')
plt.grid(True)
plt.legend()
plt.show()
