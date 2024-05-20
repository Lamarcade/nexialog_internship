# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:33:25 2024

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

#%% Create a target variable

# Cluster technique
SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 10)

SMK = SM.kmeans()
SMG, taumax = SM.classify_gaussian_mixture()

# ESG Target variables
ESGTV = SM.make_score(SMK, n_classes = 10)
#ESGTV2 = SM.make_score(SMG, n_classes = 10)

# Worst score approach
ESGTV3 = SG.worst_score(scores_ranks, n_classes = 10)
ESGTV4 = SG.worst_score(scores_ranks, n_classes = 10, reverse = True)

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
#standard_scores = SG_agencies.standardise_df()

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
 