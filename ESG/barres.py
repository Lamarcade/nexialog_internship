# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:36:23 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

from scores_utils import *

plt.close('all') 

#%% Retrieve dataframes

MS, SU, SP, RE = get_scores()
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)
scores = get_score_df()
scores_valid, valid_indices = keep_valid()
std_scores = standardise_df()
scores_ranks = scores_valid.copy()
scores_ranks = scores_ranks.rank()

# All the agencies except MSCI
dict_agencies = {'MSCI': MSS['Score'], 'SU': SUS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score']}

#triplet = std_scores[:,1:4]

#triplet_df = pd.DataFrame(triplet, columns = dict_agencies.keys())

#%% 
scale = np.arange(100)

def mapping(score, params):
    letters = ['CCC','B', 'BB', 'BBB','A','AA','AAA']
    scores = list(range(7))
    
    for i in range(len(params)):
        if score <= params[i]:
            return(scores[i])
    return 6


def scale_correl(params, MSS,SPS,RES):
    SP2, RE2 = SPS.copy(), RES.copy()
    SP2['Score'] = SP2['Score'].apply(lambda x: mapping(x, params))
    RE2['Score'] = RE2['Score'].apply(lambda x: mapping(x, params))
    
    MS2 = MSS.iloc[:-1]
    SP2 = SP2.iloc[:-1]
    
    tau1, _ = kendalltau(MS2, SP2)
    tau2, _ = kendalltau(MS2, RE2)
    
    return (tau1 + tau2) / 2

def optimize_scale(MSS,SPS,RES):
    objective = lambda params: -scale_correl(params, MSS,SPS,RES)
    init_guess = [10 + int(100 * i / 7) for i in range(7)]
    bounds = [(0.0, 100.0) for _ in range(7)]

    result = minimize(objective, init_guess, method='trust-constr', bounds=bounds)
    
    return result