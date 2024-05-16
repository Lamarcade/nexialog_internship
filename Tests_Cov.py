# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:43:02 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as linalg
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

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
#standard_scores = SG_agencies.standardise_df()
#min_max_scores = SG_agencies.min_max_df()

#SG_agencies.plot_distributions(scores_valid, "")
#SG_agencies.plot_distributions(min_max_scores, "min_max")

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
    
#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns(n_valid = 50, drop_index = 60)
#st.compute_monthly_returns(n_valid = 50)

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Su'
_ = st.keep_common_tickers(agencies_df_list[1], sectors_list)
#_ = st.keep_common_tickers(ESGTV, sectors_list)

stocks_sectors, stocks_ESG = st.select_assets(5)
#stocks_ESG = st.restrict_assets(100)
st.compute_mean()
st.compute_covariance(bias = False)
old_cov = st.get_covariance()
#st.compute_covariance(bias = True)
#cov = st.covariance_approximation()
#bias_cov = st.get_covariance()

eigv = linalg.eigvalsh(old_cov)
#bias_eigv = linalg.eigvalsh(bias_cov)

neg1 = eigv[eigv <=0]
#neg2 =  bias_eigv[bias_eigv <=0]
print(neg1)
#print(neg2)
