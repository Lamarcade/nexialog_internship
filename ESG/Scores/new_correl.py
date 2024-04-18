# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:38:19 2024

@author: LoïcMARCADET
"""

# %% Data import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import venn
from scipy.stats import spearmanr, pearsonr

from scores_utils import *

plt.close('all')

#%% Retrieve dataframes

MS, SU, SP, RE = get_scores()
#MSH, SUH, SPH, REH = homogen_df(MS, SU, SP, RE)
#MSS, SUS, SPS, RES = reduced_mixed_df(MS, SU, SP, RE)
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)

# All the agencies
dict_agencies = {'MS':  MSS['Score'], 'SU': SUS['Score'], 'SP': SPS['Score'],
    'RE': RES['Score']}
agencies_order = list(dict_agencies.keys())

scores = get_score_df(dict_agencies)
scores_valid, valid_indices = keep_valid(scores)
std_scores = standardise_df(scores_valid)

#%% Compute correlations

def get_correlations(scores = scores_valid, correl_fn = spearmanr, fn_has_statistic = True, dic = dict_agencies, aorder = agencies_order):   
    n_agencies = len(dic)
    correlations = []
    for i in range(n_agencies):
        for j in range(n_agencies):
            if j > i:
                ag1, ag2 = aorder[i], aorder[j]
                if fn_has_statistic:
                    corr = correl_fn(scores[ag1], scores[ag2]).statistic
                else:
                    corr = correl_fn(scores[ag1], scores[ag2])
                correlations.append((ag1,ag2,corr))
    return correlations

#%% Spearman correlations

spearmans = get_correlations()

#%% Uniform intervals
ui = scores_valid.copy()
ui['MS'] = ui['MS'].astype(int)
ui['SU'] = pd.qcut(ui['SU'], 9, labels = range(9))
ui['SP'] = pd.qcut(ui['SP'], 9, labels = range(9))
ui['RE'] = pd.qcut(ui['RE'], 9, labels = range(9))

spearmansUI = get_correlations(ui)

#%% Pearson on ranks 

scores_ranks, uir = scores_valid.copy(), ui.copy()
scores_ranks = scores_ranks.rank()
uir = uir.rank()

pearsonS = get_correlations(scores_ranks, pearsonr)
pearsonU = get_correlations(uir, pearsonr)
