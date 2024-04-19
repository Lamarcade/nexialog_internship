# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:29:40 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from ESG.scores_utils import *

plt.close('all') 

#%% Retrieve dataframes

MS, SU, SP, RE = get_scores('ESG/Scores/')
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)

# All the agencies
dict_agencies = {'MSCI': MSS['Score'],'SU': SUS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score']}

scores = get_score_df(dict_agencies)
scores_valid, valid_indices = keep_valid(scores)
std_scores = standardise_df(scores)
scores_ranks = scores_valid.copy()
scores_ranks = scores_ranks.rank()

#%% Clustering methods

# Class proportions
class_proportions = {'AAA': 0.093418, 'AA': 0.318471, 'A': 0.335456, 'BBB': 0.180467, 'BB': 0.061571, 'B': 0.008493, 'CCC': 0.002123}
n_classes = len(class_proportions)

def cluster_hierarchical(score_df = scores_ranks, indices = valid_indices, n_classes = 7):
    Agg = AgglomerativeClustering(n_clusters = n_classes)
        
    labels = pd.DataFrame(Agg.fit_predict(score_df), index = indices)    

    full_scores = pd.DataFrame(score_df, index = indices, columns = dict_agencies.keys())
    full_scores['labels'] = labels
    return(full_scores)

def kmeans(score_df = scores_ranks, indices = valid_indices, n_classes = 7):
    km = KMeans(n_clusters = n_classes)
        
    labels = pd.DataFrame(km.fit_predict(score_df), index = indices)    

    full_scores = pd.DataFrame(score_df, index = indices, columns = dict_agencies.keys())
    full_scores['labels'] = labels
    return(full_scores)

def classify_gaussian_mixture(score_df = scores_ranks, indices = valid_indices, n_classes = 7, class_proportions = class_proportions, n_mix = 50):
    if not(class_proportions is None):
        # Calculate weights for each cluster based on class proportions
        weights = np.array(list(class_proportions.values()))
        
        # Ensure weights are normalized according to scikit
        weights[-1] = 1-sum(weights[:-1])
    else:
        weights = None
        
    mixtures = []
    taumax = -1
    best_index = 0
    
    for i in range(n_mix): 
        mixtures.append(GaussianMixture(n_components = n_classes, weights_init = weights))
        
        mix_labels = mixtures[i].fit_predict(score_df)
    
        tau = 0
        for agency in dict_agencies:
            tau += kendalltau(dict_agencies[agency][valid_indices], mix_labels, variant = 'c').statistic / len(dict_agencies)

        if tau >= taumax:
            taumax = tau
            best_index = i

    mixture_labels = pd.DataFrame(mixtures[best_index].predict(score_df), index = indices)
    
    full_scores = pd.DataFrame(score_df, index = indices, columns = dict_agencies.keys())
    full_scores['labels'] = mixture_labels
    return full_scores, taumax

#%% Get classes

clusters = cluster_hierarchical(n_classes = 7)
kclusters = kmeans(n_classes = 7)
gaussian_classes, taumax = classify_gaussian_mixture(n_classes = 7, class_proportions=None, n_mix = 50)

#%% Make a target variable

mean_ranks = gaussian_classes.groupby('labels').mean()
global_mean_ranks = mean_ranks.mean(axis = 1)
global_mean_ranks = global_mean_ranks.sort_values()

sorted_labels = global_mean_ranks.index.tolist()  # Get the sorted cluster labels

# Create a mapping dictionary from original labels to sorted labels
label_mapping = {label: rank for rank, label in enumerate(sorted_labels, start=1)}

# Map the labels in the original DataFrame according to the sorted order
gaussian_classes['sorted_labels'] = gaussian_classes['labels'].map(label_mapping)

ESGTV = pd.DataFrame({'Tag':MSS['Tag'],'Score':gaussian_classes['sorted_labels']})

ESGTV.dropna(inplace = True)
