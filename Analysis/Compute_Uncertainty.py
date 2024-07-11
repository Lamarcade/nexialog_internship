# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:51:47 2024

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

from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, norm

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

#%% Load trained model
load_SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_SM.load_model('kmeans.pkl')
k_scores = load_SM.get_predictions()
ESGTV3 = load_SM.make_score_2(k_scores, n_classes = 7, gaussian = False)

load_GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_GSM.load_model('gauss.pkl')

full_scores = load_GSM.get_predictions()
#ESGTV4 = load_GSM.make_score_2(full_scores, n_classes = 7, gaussian = True)

ESGTV5, mean_ranks, left_inc, right_inc = load_GSM.score_uncertainty(full_scores, eta = 1)
ESGTV6, _, left_inc2, right_inc2 = load_GSM.score_uncertainty(full_scores, eta = 2)

tauLC = 0
tauLG = 0
for agency in dict_agencies:
    tauLC += kendalltau(scores_ranks[agency][valid_indices], load_SM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)

    tauLG += kendalltau(scores_ranks[agency], load_GSM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    
#%%
min_ranks, max_ranks = mean_ranks - left_inc , mean_ranks + right_inc
tri_ranks = [min_ranks, max_ranks, mean_ranks]

#load_GSM.plot_rank_uncer(tri_ranks, save = True)

#%% CDF quantiles

roots = load_GSM.quantiles_mixture()
esg_95 = np.maximum(roots, np.zeros(len(roots)))

#load_GSM.plot_rank_uncer([esg_95, mean_ranks, mean_ranks], eng = False)

list1, list2 = (list(t) for t in zip(*sorted(zip(esg_95, mean_ranks))))
list3, list4 = (list(t) for t in zip(*sorted(zip(mean_ranks, esg_95))))

#load_GSM.plot_rank_uncer([list4, list3, list3], eng = False)

#%% Expected rank

load_GSM.set_cluster_parameters()
densities, means, stds = load_GSM.get_cluster_parameters()

true_mean_ranks = scores_ranks.mean(axis = 1)

def cdf_sum(x, weights, means, stds):
    return sum(weights * norm.cdf(x, means, stds))

n, k = densities.shape
expected_ranks = np.zeros(n)
for i in range(n):
    #f = lambda x: cdf_sum(x, densities[i], means, stds)
    #rank_transfo[i] = cdf_sum(true_mean_ranks.values[i], densities[i], means, stds)
    expected_ranks[i] = sum(densities[i]*means)
   
sns.set_theme()
rank_differences = expected_ranks - true_mean_ranks
plt.figure()
sns.boxplot(rank_differences)
plt.title("Difference between expected rank from the GMM and true mean rank")