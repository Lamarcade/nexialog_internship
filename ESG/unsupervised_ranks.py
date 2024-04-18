# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:07:41 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
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
scores_ranks = scores_valid.copy()
scores_ranks = scores_ranks.rank()

# All the agencies except MSCI
dict_agencies = {'SU': SUS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score']}

#triplet = std_scores[:,1:4]

#triplet_df = pd.DataFrame(triplet, columns = dict_agencies.keys())

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

#%% Plot classes

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
plt.plot()
sns.histplot(data = gaussian_classes, x = 'labels', hue = 'labels', palette = cmap, legend = False)

plt.plot()
s = sns.pairplot(gaussian_classes, hue = 'labels', corner = True)
plt.savefig("Figures/gmm_classes.png")

plt.plot()
sns.pairplot(clusters, hue = 'labels', corner = True)
plt.savefig("Figures/h_clusters.png")

plt.plot()
sns.pairplot(kclusters, hue = 'labels', corner = True)
plt.savefig("Figures/k_clusters.png")


#%% Kendall tau for clusters

tauC = 0
for agency in dict_agencies:
    tauC += kendalltau(dict_agencies[agency][valid_indices], kclusters['labels'], variant = 'c').statistic / len(dict_agencies)
    
tauH = 0
for agency in dict_agencies:
    tauH += kendalltau(dict_agencies[agency][valid_indices], clusters['labels'], variant = 'c').statistic / len(dict_agencies)
    
#%% Ranks

gauss_ranks = gaussian_classes.copy()

# Calculate rank of scores within each label class

def add_mean_rank(df = gauss_ranks, agencies = ['SU','SP','RE'], labels_column = 'labels'):
    for agency in agencies:
        name = 'rank_' + agency
        df[name] = gauss_ranks[agency].rank()
        mr = df[[name, labels_column]].groupby(labels_column).mean().to_dict()[name]
        df['mean_'+name] = df[labels_column].map(mr)
    return None

def get_mean_ranks(df = gauss_ranks, agencies = ['SU','SP','RE'], labels_column = 'labels'):
    rank_df = pd.DataFrame(columns = agencies)
    dfc = df.copy()
    for agency in agencies:
        dfc['rank'] = gauss_ranks[agency].rank()
        mr = dfc[['rank', labels_column]].groupby(labels_column).mean().to_dict()['rank']
        rank_df[agency] = mr
        rank_df['cluster_mean_rank'] = rank_df.mean(axis=1)
    return rank_df

def get_std_ranks(df = gauss_ranks, agencies = ['SU','SP','RE'], labels_column = 'labels'):
    rank_df = pd.DataFrame(columns = agencies)
    dfc = df.copy()
    for agency in agencies:
        dfc['rank'] = gauss_ranks[agency].rank()
        mr = dfc[['rank', labels_column]].groupby(labels_column).std().to_dict()['rank']
        rank_df[agency] = mr
        rank_df['cluster_mean_rank'] = rank_df.mean(axis=1)
    return rank_df

mean_ranks = get_mean_ranks()
std_ranks = get_std_ranks()

#%%
