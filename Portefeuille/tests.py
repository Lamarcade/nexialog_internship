# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:31:01 2024

@author: LoïcMARCADET
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Portfolio import Portfolio

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

rf = 0.03 # Risk-free rate

pf = Portfolio(mu, sigma, rf)

risks, returns, sharpes = pf.efficient_frontier()

tangent_weights = pf.tangent_portfolio()
tangent_risk, tangent_return = np.sqrt(pf.get_variance(tangent_weights)), pf.get_return(tangent_weights)


pf.risk_free_stats()

risks_rf, returns_rf, _ = pf.efficient_frontier(max_risk_tol = 0.5)

tangent_weights_rf = pf.tangent_portfolio()
tangent_risk_rf, tangent_return_rf = np.sqrt(pf.get_variance(tangent_weights_rf)), pf.get_return(tangent_weights_rf)


#%% Plotting the efficient frontier

plt.figure(figsize=(8, 6))
plt.scatter(risks, returns, c=sharpes, cmap='viridis', s = 1)
plt.colorbar(label='Sharpe Ratio')
plt.plot(risks_rf, returns_rf, linestyle = '--')
plt.plot(tangent_risk, tangent_return, marker='o', color='r', markersize=5, label = "Tangent Portfolio")

plt.title('Markowitz Efficient Frontier with and without a risk-free asset')
plt.xlabel('Portfolio Risk')
plt.ylabel('Portfolio Return')
plt.grid(True)
plt.legend()
plt.show()