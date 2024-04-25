# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:46:33 2024

@author: LoïcMARCADET
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker

path = "Portefeuille/sp500_stocks_short.csv"
annual_rf = 0.05 # Risk-free rate


#%% Retrieve the scores and compute the ranks 
SG = ScoreGetter('ESG/Scores/')
SG.reduced_mixed_df()
score_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

#%% Create a target variable
SM = ScoreMaker(score_ranks, dict_agencies, valid_tickers, valid_indices, 7)

SMK = SM.kmeans()

ESGTV = SM.make_score(SMK)

#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()
_ = st.keep_common_tickers(ESGTV)
stocks_ESG = st.restrict_assets(20)
st.compute_mean()
st.compute_covariance()
mean, cov, rf = st.get_mean(), st.get_covariance(), st.get_rf()


#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales= False)
epf = epf.risk_free_stats()

sharpes, ESG_list = epf.efficient_frontier_ESG(min(stocks_ESG), max(stocks_ESG) + 1, interval = 1)

plt.figure(figsize=(8, 6))
plt.plot(ESG_list, sharpes, label='Efficient Frontier', marker='o', linestyle='-')
plt.title('ESG Constraints impact on Sharpe Ratio')
plt.xlabel('ESG score')
plt.ylabel('Sharpe ratio')
plt.grid(True)
plt.legend()
plt.show()

#%% 
risks, returns, sharpes = epf.efficient_frontier(method = 2)
risks_5, returns_5, sharpes_5 = epf.efficient_frontier(method = 2, new_constraints = [epf.ESG_constraint(5)])

plt.figure(figsize=(8, 6))
plt.plot(risks, returns)
plt.plot(risks_5, returns_5, label = 'Min ESG of 5')
plt.title('ESG Constraints impact on Efficient frontier')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.grid(True)
plt.legend()
plt.show()