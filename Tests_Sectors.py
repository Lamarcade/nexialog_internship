# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:54:24 2024

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
ESGTV3 = SG.worst_score(scores_ranks, n_classes = 10)

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
scores_valid = SG_agencies.keep_valid()

agencies_df_list = []
for agency in scores_valid.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
    

#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Worst'

_ = st.keep_common_tickers(ESGTV3, sectors_list)

#n_assets = 50
#stocks_ESG = st.restrict_assets(n_assets)

stocks_sectors, stocks_ESG = st.select_assets(5)

st.compute_mean()
st.compute_covariance()
mean, _, rf = st.get_mean(), st.get_covariance(), st.get_rf()
cov = st.covariance_approximation()

st.plot_sectors()

#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = sectors_list.loc[st.index.values])

#epf = epf.risk_free_stats()

#%% Efficient frontier depending on the sector minimum constraint

risks, returns, sharpes = epf.efficient_frontier(max_std = 0.10, method = 1)
epf.new_figure()
#epf.plot_tangent(tangent_risk, tangent_return)
epf.plot_constrained_frontier(risks, returns)

save = False

weight_10_range = [0.01, 0.02, 0.05, 0.09]
weight_50_range = [0.001, 0.005, 0.01, 0.015]

count, num_iters = 1, 4


for bound in weight_10_range:
    print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = num_iters))
    risks_new, returns_new, sharpes_new = epf.efficient_frontier(max_std = 0.10, method = 1, new_constraints = [epf.sector_constraint(bound)])
    if count == 4:
        save = True
    epf.plot_constrained_frontier(risks_new, returns_new, sector_min = bound, title = "_min_sectors_", savefig = save, score_source = provider)
    count += 1
    
#%% Efficient frontier depending on the sector maximum constraint

risks, returns, sharpes = epf.efficient_frontier(max_std = 0.10, method = 1)
epf.new_figure()
#epf.plot_tangent(tangent_risk, tangent_return)
epf.plot_constrained_frontier(risks, returns)

save = False
max_10_range = [0.9, 0.5, 0.2, 0.15]
max_50_range = [0.9, 0.1, 0.05, 0.025]

count, num_iters = 1, 4

for bound in max_10_range:
    print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = num_iters))
    risks_new, returns_new, sharpes_new = epf.efficient_frontier(max_std = 0.10, method = 1, new_constraints = [epf.sector_constraint(bound, is_min = False)])
    if count == 4:
        save = True
    epf.plot_constrained_frontier(risks_new, returns_new, sector_max = bound, title = '_max_sectors_', savefig = save, score_source = provider)
    count += 1