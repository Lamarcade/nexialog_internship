# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:40:29 2024

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

# Worst score approach
ESGTV3, all_ranks = SG.worst_score(scores_ranks, n_classes = 7, get_all = True)
ESGTV4 = SG.worst_score(scores_ranks, n_classes = 7, reverse = True)

ESGTV5 = pd.DataFrame({'Tag': ESGTV3['Tag'], 'Score': round(all_ranks.mean(axis = 1)).astype(int)})

range_number = range(len(ESGTV3))
# =============================================================================
# plt.figure(figsize = (20,6))
# plt.plot(range_number, ESGTV3['Score'], 'bo', label = 'Worst')
# plt.plot(range_number, ESGTV4['Score'], 'go', label = 'Best')
# plt.plot(range_number, ESGTV5['Score'], 'ro', label = 'Average')
# plt.legend()
# plt.show()
# =============================================================================
esg_df = pd.DataFrame({'Tag': valid_tickers, 'Worst': ESGTV3['Score'], 'Best': ESGTV4['Score'], 'Mean': ESGTV5['Score']})

#dist_df = pd.DataFrame({'Worst': ESGTV3['Score'], 'Best': ESGTV4['Score'], 'Mean': ESGTV5['Score']})
dist_df = pd.DataFrame({'Pire': ESGTV3['Score'], 'Meilleur': ESGTV4['Score'], 'Moyen': ESGTV5['Score']})
SG.plot_distributions(dist_df, dist_type = 'harmonisés', shrink = 0.5, n = 3, eng = False)

#%%
SG2 = ScoreGetter('ESG/Scores/')
SG2.reduced_df()
SG2.set_valid_df()
std_df = SG2.standardise_df()

# Worst score approach
ESGTV6, all_scores = SG2.worst_std_score(std_df, n_classes = 7, get_all = True)
ESGTV7 = SG2.worst_std_score(std_df, n_classes = 7, reverse = True)

ESGTV8 = pd.DataFrame({'Tag': ESGTV6['Tag'], 'Score': round(all_scores.mean(axis = 1)).astype(int)})

range_number = range(len(ESGTV6))

#dist_df = pd.DataFrame({'Worst': ESGTV3['Score'], 'Best': ESGTV4['Score'], 'Mean': ESGTV5['Score']})
dist_df = pd.DataFrame({'Pire': ESGTV6['Score'], 'Meilleur': ESGTV7['Score'], 'Moyen': ESGTV8['Score']})
SG.plot_distributions(dist_df, dist_type = 'standardisés puis harmonisés', shrink = 0.5, n = 3, eng = False)