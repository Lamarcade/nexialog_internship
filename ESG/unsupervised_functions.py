# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:32:25 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import gower
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from scores_utils import *

plt.close('all') 

#%% Retrieve dataframes

MS, SU, SP, RE = get_scores()
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)
scores = get_score_df()
scores_valid, valid_indices = keep_valid()
std_scores = standardise_df()

# All the agencies
dict_agencies = {'MS':  MSS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score'], 'SU': SUS['Score']}

#%% Clustering methods

# Class proportions
class_proportions = {'AAA': 0.093418, 'AA': 0.318471, 'A': 0.335456, 'BBB': 0.180467, 'BB': 0.061571, 'B': 0.008493, 'CCC': 0.002123}
n_classes = len(class_proportions)

def cluster_hierarchical(score_df = std_scores, indices = valid_indices, n_classes = 7):
    Agg = AgglomerativeClustering(n_clusters = n_classes)
        
    labels = pd.DataFrame(Agg.fit_predict(score_df), index = valid_indices)    

    full_scores = pd.DataFrame(score_df, index = indices)
    full_scores['labels'] = labels
    return(full_scores)

def kmeans(score_df = std_scores, indices = valid_indices, n_classes = 7):
    km = KMeans(n_clusters = n_classes)
        
    labels = pd.DataFrame(km.fit_predict(score_df), index = indices)    

    full_scores = pd.DataFrame(score_df, index = indices)
    full_scores['labels'] = labels
    return(full_scores)

def classify_gaussian_mixture(score_df = std_scores, indices = valid_indices, n_classes = 7, class_proportions = class_proportions, n_mix = 50):
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
    
    full_scores = pd.DataFrame(score_df, index = indices)
    full_scores['labels'] = mixture_labels
    return full_scores, taumax

#%% Get classes

clusters = cluster_hierarchical(n_classes = 7)
kclusters = kmeans(n_classes = 7)
gaussian_classes, taumax = classify_gaussian_mixture(n_classes = 7, n_mix = 50)

#%% Plot classes

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
plt.plot()
sns.histplot(data = gaussian_classes, x = 'labels', hue = 'labels', palette = cmap, legend = False)

plt.plot()
sns.pairplot(gaussian_classes, hue = 'labels', corner = True)

plt.plot()
sns.pairplot(clusters, hue = 'labels', corner = True)

plt.plot()
sns.pairplot(kclusters, hue = 'labels', corner = True)

#%% Kendall tau for clusters

tauC = 0
for agency in dict_agencies:
    tauC += kendalltau(dict_agencies[agency][valid_indices], kclusters['labels'], variant = 'c').statistic / len(dict_agencies)