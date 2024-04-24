# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:24:36 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Portfolio import Portfolio
from Stocks import Stocks

path = "Portefeuille/sp500_stocks_short.csv"
annual_rf = 0.05 # Risk-free rate

st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()
st.restrict_assets(20)
st.compute_mean()
st.compute_covariance()
mean, cov, rf = st.get_mean(), st.get_covariance(), st.get_rf()
#%%

pf = Portfolio(mean, cov, rf)

risks, returns, sharpes = pf.efficient_frontier(method = 3, max_risk_tol = 0.5, max_std = 0.08)

tangent_weights = pf.tangent_portfolio()
tangent_risk, tangent_return = np.sqrt(pf.get_variance(tangent_weights)), pf.get_return(tangent_weights)

pf = pf.risk_free_stats()

risks_rf, returns_rf, _ = pf.efficient_frontier(method = 3, max_risk_tol = 0.5, max_std = 0.08)

#%% Plotting the efficient frontier
pf.new_figure()
pf.plot_frontier(False, risks, returns, sharpes = sharpes)
pf.plot_frontier(True, risks_rf, returns_rf)
pf.plot_tangent(tangent_risk, tangent_return)

#%% Comparing with the no short sales restriction
pf.new_figure()
pf.plot_frontier(True, risks_rf, returns_rf)
pf = pf.change_short_sales()
risks_ns, returns_ns, _ = pf.efficient_frontier(method = 2, max_risk_to = 0.5, max_std = 0.08)
pf.plot_frontier(True, risks_ns, returns_ns)
pf.plot_tangent(tangent_risk, tangent_return)
