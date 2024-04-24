# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:29:16 2024

@author: LoïcMARCADET
"""

#%% Libraries  
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

#%%


class ScoreMaker:
    def __init__(self, ranks, dict_agencies, valid_tickers, valid_indices, n_classes):
        self.ranks = ranks
        self.dict_agencies = dict_agencies
        self.valid_tickers = valid_tickers
        self.valid_indices = valid_indices
        self.n_classes = n_classes
    
        
    def add_class(self, labels):
        full_scores = pd.DataFrame(self.ranks, index = self.valid_indices)
        full_scores['labels'] = labels
        return(full_scores)
    
    def cluster_hierarchical(self):
        Agg = AgglomerativeClustering(n_clusters = self.n_classes)
            
        labels = pd.DataFrame(Agg.fit_predict(self.ranks), index = self.valid_indices)   

        return(self.add_class(labels))
    

    def kmeans(self):
        km = KMeans(n_clusters = self.n_classes)
            
        labels = pd.DataFrame(km.fit_predict(self.ranks), index = self.valid_indices)    
        return(self.add_class(labels))

    def classify_gaussian_mixture(self, class_proportions = None, n_mix = 50):
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
            mixtures.append(GaussianMixture(n_components = self.n_classes, weights_init = weights))
            
            mix_labels = mixtures[i].fit_predict(self.ranks)
        
            tau = 0
            for agency in self.dict_agencies:
                #tau += kendalltau(self.dict_agencies[agency][self.valid_tickers], mix_labels, variant = 'c').statistic / len(dict_agencies)
                tau += -kendalltau(self.ranks, mix_labels, variant = 'c').statistic

            if tau >= taumax:
                taumax = tau
                best_index = i

        mixture_labels = pd.DataFrame(mixtures[best_index].predict(self.ranks), index = self.valid_indices)
        
        full_scores = self.add_class(mixture_labels)
        return full_scores, taumax
    
    def make_score(self, full_scores):
        mean_ranks = full_scores.groupby('labels').mean()
        global_mean_ranks = mean_ranks.mean(axis = 1)
        global_mean_ranks = global_mean_ranks.sort_values()

        sorted_labels = global_mean_ranks.index.tolist()  # Get the sorted cluster labels
        
        # Create a mapping dictionary from original labels to sorted labels
        label_mapping = {label: rank for rank, label in enumerate(sorted_labels, start=1)}

        # Map the labels in the original DataFrame according to the sorted order
        full_scores['sorted_labels'] = full_scores['labels'].map(label_mapping)

        ESGTV = pd.DataFrame({'Tag':self.valid_tickers,'Score':full_scores['sorted_labels']})

        ESGTV.dropna(inplace = True)
        return ESGTV