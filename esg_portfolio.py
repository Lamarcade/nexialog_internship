# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:11:28 2024

@author: LoïcMARCADET
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd

from ESG.scores_utils import * 
from Portefeuille.portfolio_utils import *
from make_scores import *

#%% 
stocks = pd.read_csv("Portefeuille/sp500_stocks_short.csv")

# Convert 'Date' column to datetime
stocks['Date'] = pd.to_datetime(stocks['Date'])

# Pivot the DataFrame to have symbols as columns
stocks_pivot = stocks.pivot(index='Date', columns='Symbol', values='Adj Close')

# Calculate monthly returns for each symbol
monthly_returns = stocks_pivot.resample('ME').ffill().pct_change()

monthly_returns = monthly_returns.iloc[1:]

# Drop columns that have at least 50 NaN
monthly_returns = monthly_returns.dropna(axis=1, thresh=50)

#%% Get ESG variable
target_variable = ESGTV

#%% Filter out missing values

#Keep only the companies for which we have monthly returns
mr_tickers = monthly_returns.columns.tolist()
tv_filtered = target_variable.loc[target_variable['Tag'].isin(mr_tickers)]

# Keep only the companies for with we have ESG Scores
tv_tickers = tv_filtered['Tag'].tolist()
mr_filtered = monthly_returns.loc[:, monthly_returns.columns.isin(tv_tickers)]

#%%
n_assets = 200
mr = mr_filtered.iloc[:, :n_assets]

means = np.array(mr.mean(axis = 0))
cov = np.array(mr.cov())


ESG_tv = np.array(tv_filtered['Score'].tolist()[:n_assets])
#%%
r_annual = 0.05 # Risk-free rate
rf = (1+r_annual)**(1/12)-1

means_rf = rf_mean(means, rf)
cov_rf = rf_var(cov)
# ESG scores of the assets
ESGs = np.random.random(n_assets)

# ESG preference functions
f = lambda s: s/10
g = lambda s: s+1

#%%
# Standard efficient frontier without a risk-free asset
ef_points = efficient_jac(means, cov, rf, min_risk_tol=0, num_points = 100, short_sales = True)
stds, mus = [p[0] for p in ef_points], [p[1] for p in ef_points]
sh = get_sharpe(mus, stds, rf)

# Standard efficient frontier with a risk-free asset
ef_points_rf = efficient_jac(means_rf, cov_rf, rf, min_risk_tol = 0, num_points = 100, short_sales = True)
stds_rf, mus_rf = [p[0] for p in ef_points_rf], [p[1] for p in ef_points_rf]
sh_rf = get_sharpe(mus_rf, stds_rf, rf)

# Risk and return of the optimal portfolio
tangent_std, tangent_ret = optim_sharpe(means, cov, rf)
tangent_std_rf, tangent_ret_rf = optim_sharpe(means_rf, cov_rf, rf)

std_range = np.arange(0.001, 0.20, 0.01)
cml = capital_market_line(rf,tangent_ret, tangent_std, std_range)

# =============================================================================
# # Dirichlet distribution for the frontier close to the origin    
# dir_avg = np.ones(n_assets+1)
# dir_avg[0] = n_assets
# 
# # Dirichlet distribution for the frontier in the upper-right part
# profit_idx = np.argmax(means_rf)
# dir0_avg = np.ones(n_assets+1)
# dir0_avg[profit_idx] = n_assets
# 
# # Random weights to verify the efficient frontier
# random_sigma, random_mu = random_weights(n_assets, means_rf, cov_rf, method = 'rand', dir_alpha = None, n_samples = 500)
# dir_sigma, dir_mu = random_weights(n_assets, means_rf, cov_rf, method = 'dirichlet', dir_alpha = dir_avg, n_samples = 500)
# dir0_sigma, dir0_mu = random_weights(n_assets, means_rf, cov_rf, method = 'dirichlet', dir_alpha = dir0_avg, n_samples = 500)
# =============================================================================

#%% Plotting the efficient frontier
plt.figure(figsize=(8, 6))
#plt.plot(stds, mus, label='Efficient Frontier', marker='o', linestyle='-')
plt.scatter(stds, mus, c=sh, cmap='viridis', label = 'efficient frontier')
plt.colorbar(label='Sharpe Ratio')
plt.plot(stds_rf, mus_rf, label='CML', linestyle='--')
plt.plot(tangent_std, tangent_ret, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.plot(tangent_std_rf, tangent_ret_rf, marker='o', color='g', markersize=5, label = "TP with a risk-free asset")
#plt.plot(std_range,cml, label = "CML", linestyle = "--")


#plt.scatter(random_sigma, random_mu, s=0.1, color='g', label = "Random weights")
#plt.scatter(dir_sigma, dir_mu, s=0.1, color='g', label = "Dirichlet with weight 30 on risk-free")
#plt.scatter(dir0_sigma, dir0_mu, s=0.1, color='g', label = "Dirichlet with weight 30 on highest return")
#plt.scatter(stdse, muse, c=she)
#plt.plot(stdsa, musa, label='Efficient Frontier with ' + str(low_ESG) + ' <= ESG <= ' + str(up_ESG), marker='o', linestyle='-')
plt.title('Markowitz Efficient Frontier and Capital Market Line')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()

#%% Draw the maximum Sharpe ratio depending on the ESG constraints

stds_c, mus_c, min_esg_list = [],[],[]

# We consider only ESG constraints feasible for the given companies
for lESG in np.arange(min(ESG_tv),max(ESG_tv) + 1,1):
    stp, mup, _ = optim_sharpe_esg(means_rf, cov_rf, ESG_tv, rf, lESG, 7, num_points = 100)
    stds_c.append(stp)
    mus_c.append(mup)
    min_esg_list.append(lESG)
    
sh_c = get_sharpe(mus_c, stds_c, rf)    
    
plt.figure(figsize=(8, 6))
plt.plot(min_esg_list, sh_c, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('ESG Constraints impact on Sharpe Ratio')
plt.xlabel('ESG score')
plt.ylabel('Sharpe ratio')
plt.grid(True)
plt.legend()
plt.show()
