# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:03:58 2024

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

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
standard_scores = SG_agencies.standardise_df()
#min_max_scores = SG_agencies.min_max_df()

#SG_agencies.plot_distributions(scores_valid, "no_norma")
#SG_agencies.plot_distributions(min_max_scores, "min_max")

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_ranks[agency]}))
    
#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Re'
_ = st.keep_common_tickers(agencies_df_list[3], sectors_list)
#_ = st.keep_common_tickers(ESGTV, sectors_list)

stocks_sectors, stocks_ESG = st.select_assets(5)
#stocks_ESG = st.restrict_assets(50)
st.compute_mean()
st.compute_covariance()
mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
cov = st.covariance_approximation()

#st.plot_sectors()

#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, tickers = stocks_sectors['Tag'].tolist())
#tangent_weights = epf.tangent_portfolio()
#tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)

#epf = epf.risk_free_stats()


#sharpes, ESG_list = epf.efficient_frontier_ESG(0, 332, interval = 10)

#epf.plot_ESG_frontier(sharpes, ESG_list, savefig = True, score_source = provider)

#%% Efficient frontier depending on the ESG constraint

risks, returns, sharpes = epf.efficient_frontier(max_std = 0.10, method = 2)
epf.new_figure()
#epf.plot_tangent(tangent_risk, tangent_return)
epf.plot_constrained_frontier(risks, returns)

save = False
step = 40
emin, emax = 130, 332
#count, num_iters = 1, 1 + ((int(max(stocks_ESG))) - int(min(stocks_ESG))) // step
count, num_iters = 1, 1 + (emax - emin) // step

for min_ESG in range(emin, emax, step):
    if not(count % 2):
        print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = int(num_iters)))
    risks_new, returns_new, sharpes_new = epf.efficient_frontier(max_std = 0.10, method = 2, new_constraints = [epf.ESG_constraint(min_ESG)])
    if min_ESG >= (emax-1-step):
        save = True
    epf.plot_constrained_frontier(risks_new, returns_new, ESG_min_level = min_ESG, savefig = save, score_source = provider, eng = False)
    count += 1

#%% Sector evolution
ESG_range = range(emin, emax, step)
    
for i,agency in enumerate(scores_ranks.columns):
    
    st = Stocks(path, annual_rf)
    st.process_data()
    st.compute_monthly_returns()

    # 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
    _ = st.keep_common_tickers(agencies_df_list[i], sectors_list)
    #_ = st.keep_common_tickers(ESGTV, sectors_list)

    stocks_sectors, stocks_ESG = st.select_assets(5)
    #stocks_ESG = st.restrict_assets(50)
    st.compute_mean()
    st.compute_covariance()
    mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
    cov = st.covariance_approximation()

    epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, tickers = stocks_sectors['Tag'].tolist())

    #epf.plot_asset_evolution(range(emin, emax, step), stocks_sectors, save = True, source = 'Refinitiv', min_weight = 0.001, assets_weights = None, xlabel = "ESG rank constraint")
    
    assets_weights = epf.get_evolution(ESG_range)
    epf.plot_asset_evolution(range(emin, emax, step), stocks_sectors, save = True, source = agency, min_weight = 0.001, assets_weights = assets_weights, xlabel = "Contrainte de rang ESG", eng = False)
    
    sectors_weights = epf.sectors_evolution_from_tickers(assets_weights, stocks_sectors)
    
    epf.plot_sector_evolution(ESG_range, save = True, source = agency, min_weight = 0.001, sectors_weights = sectors_weights, xlabel = "Contrainte de rang ESG", eng = False)

