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

sectors_list = SG.valid_sector_df

#%% Create a target variable

# Cluster technique
SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 10)

SMK = SM.kmeans()
SMG, taumax = SM.classify_gaussian_mixture()

# ESG Target variables
ESGTV = SM.make_score(SMK, n_classes = 10)
ESGTV2 = SM.make_score(SMG, n_classes = 10)

# Worst score approach
ESGTV3 = SG.worst_score(scores_ranks, n_classes = 10)

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
#standard_scores = SG_agencies.standardise_df()
#min_max_scores = SG_agencies.min_max_df()

#SG_agencies.plot_distributions(old_scores, "")
#SG_agencies.plot_distributions(min_max_scores, "min_max")

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
    
#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Su'
_ = st.keep_common_tickers(agencies_df_list[1], sectors_list)
#_ = st.keep_common_tickers(ESGTV, sectors_list)

stocks_sectors, stocks_ESG = st.select_assets(5)
#stocks_ESG = st.restrict_assets(50)
st.compute_mean()
st.compute_covariance()
mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
cov = st.covariance_approximation()

st.plot_sectors()

#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False)
#tangent_weights = epf.tangent_portfolio()
#tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)

epf = epf.risk_free_stats()


sharpes, ESG_list = epf.efficient_frontier_ESG(min(stocks_ESG), max(stocks_ESG) + 1, interval = 1)

epf.plot_ESG_frontier(sharpes, ESG_list, savefig = True, score_source = provider)

#%% Efficient frontier depending on the ESG constraint

risks, returns, sharpes = epf.efficient_frontier(max_std = 0.10, method = 1)
epf.new_figure()
#epf.plot_tangent(tangent_risk, tangent_return)
epf.plot_constrained_frontier(risks, returns)

save = False
step = 1
count, num_iters = 1, 1 + ((int(max(stocks_ESG))) - int(min(stocks_ESG))) // step


for min_ESG in range(int(min(stocks_ESG)), int(max(stocks_ESG)) + step, step):
    if not(count % 2):
        print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = int(num_iters)))
    risks_new, returns_new, sharpes_new = epf.efficient_frontier(max_std = 0.10, method = 1, new_constraints = [epf.ESG_constraint(min_ESG)])
    if int(max(stocks_ESG)) - min_ESG < step:
        save = True
    epf.plot_constrained_frontier(risks_new, returns_new, ESG_min_level = min_ESG, savefig = save, score_source = provider)
    count += 1

#%% Exclude worst ESG stocks from the universe

threshold_list = np.arange(0, 1, 0.05)
sharpes_t = []
# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Su'
    
for threshold in threshold_list:
    est = Stocks(path, annual_rf)
    est.process_data()
    est.compute_monthly_returns()

    
    _ = est.keep_common_tickers(agencies_df_list[1], sectors_list)
    
    _, _ = est.select_assets(5)
    stocks_sectors, stocks_ESG = est.exclude_assets(threshold)
    
    est.compute_mean()
    est.compute_covariance()
    mean, _, rf = est.get_mean(), est.get_covariance(), est.get_rf()
    cov = est.covariance_approximation()

    xpf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = stocks_sectors)
    xpf = xpf.risk_free_stats()
    
    weights_t = xpf.tangent_portfolio()
    sharpes_t.append(xpf.get_sharpe(weights_t))

xpf.new_figure()
xpf.plot_sharpe_exclusion(sharpes_t, threshold_list, True, provider)
    
    