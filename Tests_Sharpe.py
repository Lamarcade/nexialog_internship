# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:24:41 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker

from scipy.stats import spearmanr

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
#ESGTV3 = SG.worst_score(scores_ranks, n_classes = 7)

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
min_max_scores = SG_agencies.min_max_df()

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': min_max_scores[agency]}))
    
#%% Get the stock data and keep the companies in common with the target variable

def Sharpe_analysis(df_list, dict_agencies, low_ESG = 0, up_ESG = 1.05, step = 0.05):
    
    save = False
    spearmans = {}
    # 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
    for i, agency in enumerate(dict_agencies.keys()):
        #provider = 'Su'
        st = Stocks(path, annual_rf)
        st.process_data()
        st.compute_monthly_returns()
        _ = st.keep_common_tickers(agencies_df_list[i], sectors_list)
        #_ = st.keep_common_tickers(ESGTV, sectors_list)
        
        stocks_sectors, stocks_ESG = st.select_assets(5)
        #stocks_ESG = st.restrict_assets(50)
        st.compute_mean()
        st.compute_covariance()
        mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
        cov = st.covariance_approximation()
        
        # Build a portfolio with restrictions on the minimal ESG score
        if i == 0:
            epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = stocks_sectors)
            #tangent_weights = epf.tangent_portfolio()
            #tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
        
            epf = epf.risk_free_stats()
            epf.new_figure(fig_size = (12,12))
        
        epf.set_ESGs(stocks_ESG)
        
        sharpes, ESG_list = epf.efficient_frontier_ESG(low_ESG, up_ESG, interval = step)
        spearmans[agency] = spearmanr(sharpes, ESG_list).statistic
        if agency == 'RE':
            save = True
        epf.plot_sharpe_speed(sharpes, ESG_list, save = save, source = agency)
    return spearmans, epf
        
step = 0.05
low_ESG, up_ESG = 0, 1.05
ESG_range = np.arange(low_ESG,up_ESG, step)
spearmans, epf = Sharpe_analysis(agencies_df_list, dict_agencies)

#%% 
epf.plot_sector_evolution(ESG_range, save = True, source = "Refinitiv")
epf.plot_sector_evolution(np.arange(0.7, 0.95, 0.05), save = True, source = "Refinitiv")