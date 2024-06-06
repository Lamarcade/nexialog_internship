# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:11:40 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker
import seaborn as sns
from scipy.stats import kendalltau

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
SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)

SMK = SM.kmeans()
SMG, taumax = GSM.classify_gaussian_mixture()

# ESG Target variables
ESGTV = SM.make_score_2(SMK, n_classes = 7)
ESGTV2 = GSM.make_score_2(SMG, n_classes = 7)



#%% Plot classes

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
plt.plot()
sns.histplot(data = SM.full_ranks, x = 'sorted_labels', hue = 'sorted_labels', palette = cmap, legend = False)

plt.plot()
s = sns.pairplot(SM.full_ranks[['MS', 'SU','SP','RE','sorted_labels']], hue = 'sorted_labels', corner = True)
s.fig.suptitle('Classes obtained with a Gaussian Mixture Model', y = 1.03)
plt.savefig("Figures/gmm_classes.png", bbox_inches = 'tight')

plt.plot()
ks = sns.pairplot(GSM.full_ranks[['MS', 'SU','SP','RE','sorted_labels']], hue = 'sorted_labels', corner = True)
ks.fig.suptitle('Classes obtained with a K-Means model', y = 1.03)
plt.savefig("Figures/k_clusters.png", bbox_inches = 'tight')

#%% Kendall tau for clusters

tauC = 0
for agency in dict_agencies:
    tauC += kendalltau(scores_ranks[agency][valid_indices], SM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    
#%% Rank stats

mn, st = SM.get_mean_ranks(), SM.get_std_ranks()
mng, stg = GSM.get_mean_ranks(), GSM.get_std_ranks()
