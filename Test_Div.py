# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:27:24 2024

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
SG.reduced_df()
scores_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

SG.valid_ticker_sector()

sectors_list = SG.valid_sector_df

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
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': standard_scores[agency]}))

#%% 

def DR_analysis(df_list, list_agencies, low_ESG = 0, up_ESG = 1.05, step = 0.01):
    
    save = False
    reduced_df = pd.DataFrame()
    # 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
    for i, agency in enumerate(list_agencies):
        print(i)
        #provider = 'Su'
        st = Stocks(path, annual_rf)
        st.process_data()
        st.compute_monthly_returns()
        _ = st.keep_common_tickers(df_list[i], sectors_list)
        #_ = st.keep_common_tickers(ESGTV, sectors_list)
        
        stocks_sectors, stocks_ESG = st.select_assets(5)
        #stocks_ESG = st.restrict_assets(50)
        st.compute_mean()
        st.compute_covariance()
        mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
        cov = st.covariance_approximation()
        
        reduced_df[agency] = stocks_ESG
        
        # Build a portfolio with restrictions on the minimal ESG score
        if i == 0:
            epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = stocks_sectors)
            #tangent_weights = epf.tangent_portfolio()
            #tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
        
            epf.new_figure(fig_size = (12,12))
        
        epf.set_ESGs(stocks_ESG)
        
        emin, emax = min(stocks_ESG), max(stocks_ESG)
        DRs, ESG_list = epf.diversification_ESG(emin, emax + step, interval = step)
        if agency == 'RE':
            save = True
        epf.plot_general_frontier(ESG_list, DRs, fig_label = agency, fig_title = "DRs", xlabel = "ESG constraint", ylabel = "Diversification Ratio", save = save, new_fig = False)

    return epf, reduced_df
        
step = 0.05
low_ESG, up_ESG = -1, 2
epf, reduced_df = DR_analysis(agencies_df_list, ['MS', 'SU', 'SP', 'RE'], low_ESG, up_ESG, step)
