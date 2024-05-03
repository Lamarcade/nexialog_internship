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
scores_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

SG.valid_ticker_sector()

sectors_list = SG.get_valid_sector_df()

#%% Create a target variable

# Cluster technique
#SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)

#SMK = SM.kmeans()
#SMG, taumax = SM.classify_gaussian_mixture()

# ESG Target variables
#ESGTV = SM.make_score(SMK)
#ESGTV2 = SM.make_score(SMG)

# Worst score approach
#ESGTV3 = SG.worst_score(scores_ranks, n_classes = 7)

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
_ = SG_agencies.get_score_df()
scores_valid = SG_agencies.keep_valid()

agencies_df = []
for agency in scores_valid.columns:
    agencies_df.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
    

#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Refinitiv'

_ = st.keep_common_tickers(agencies_df[1], sectors_list)
stocks_ESG = st.restrict_assets(50)
st.compute_mean()
st.compute_covariance()
mean, _, rf = st.get_mean(), st.get_covariance(), st.get_rf()
cov = st.covariance_approximation()

st.plot_sectors()

#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = sectors_list)

epf = epf.risk_free_stats()

#%% Sector weights for the tangent portfolio

#epf.plot_tangent(tangent_risk, tangent_return)

save = False
count, num_iters = 1, (int(max(stocks_ESG)) - int(min(stocks_ESG))) // 5

ESG_range = range(int(min(stocks_ESG)),int(max(stocks_ESG)) + 1, 5)

epf.new_figure()

for min_ESG in ESG_range:
    if not(count % 2):
        print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = num_iters))
    tangent_weights = epf.optimal_portfolio_ESG(min_ESG)
    tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
    epf.set_sectors_composition(tangent_weights)
    if int(max(stocks_ESG)) - min_ESG < 5:
        save = True
    epf.plot_sectors_composition(min_ESG, save, provider)
    count += 1
    
#%% Sector weights evolution depending on the ESG constraint

epf.plot_composition_change(60, 90, True, provider)
