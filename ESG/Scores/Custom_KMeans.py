# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:39:15 2024

@author: LoïcMARCADET
"""
import numpy as np
import random
from scores_utils import *
import seaborn as sns
import matplotlib.pyplot as plt

class ESG_KMeans:
    
    def __init__(self, n_clusters = 5, n_scores = 4,max_iter = 500):
        """
        Constructor for KMeans class
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iters : int
            Maximum number of iterations, by default 500
        """    
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        
    def get_ranks(self, score_df):
        """
        Compute score ranks for each agencies and average rank for each observation

        Parameters
        ----------
        X : Pandas Dataframe
            Input data
        
        Returns
        -------
        X_ranks : Pandas Dataframe
            Dataframe with the ranks and average ranks
        """
        X_ranks = score_df.copy()
        X_ranks = X_ranks.rank()
        X_ranks['rank_avg'] = X_ranks.mean(axis=1).astype(int)
        return X_ranks
    
    def sort_by_avg_rank(self, X_ranks, asc_order = True):
        """
        Sort the dataframe by increasing

        Parameters
        ----------
        X_ranks : Pandas Dataframe
            Dataframe with the ranks and average ranks
        asc_order : bool
            Indicates whether to sort by ascending (default) or decreasing order
        Returns
        -------
        X_sorted : Pandas Dataframe
            Dataframe with the ranks sorted by average rank
        """
        return X_ranks.sort_values(by = 'rank_avg', ascending = asc_order)
    
    def fit(self, X_sorted):
        """
        Perform KMeans to update the centroids
        
        Parameters
        ----------
        X_sorted : Pandas Dataframe
            Dataframe with the ranks sorted by average rank
        """    
        min_rank, max_rank = np.min(X_sorted['rank_avg']), np.max(X_sorted['rank_avg'])
        step = max_rank // self.n_clusters
        
        self.centroids = [X_sorted.iloc[i*step] for i in range(self.n_clusters)]
        
        iters, old_centroids = 0, None
        while not(np.equal(old_centroids,self.centroids).all()) and iters <= self.max_iter:
            assigned_points = [[] for _ in range(self.n_clusters)]
            for i in range(len(X_sorted)):
                x = X_sorted.iloc[i].to_numpy()
                distances = [np.linalg.norm(x-y) for y in self.centroids]
                centroid_idx = np.argmin(distances)
                assigned_points[centroid_idx].append(x)
                
            old_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis = 0) for cluster in assigned_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = old_centroids[i]
            iters +=1
           
    def predict(self, X_sorted):
        centroid_idxs = []
        
        for i in range(len(X_sorted)):
            x = X_sorted.iloc[i].to_numpy()
            distances = [np.linalg.norm(x-y) for y in self.centroids]
            centroid_idx = np.argmin(distances)
            centroid_idxs.append(centroid_idx)
        return centroid_idxs  
    
    def predict_assign(self, X_sorted):
        centroid_idxs = self.predict(X_sorted)
        df = X_sorted.copy()
        df['cluster'] = centroid_idxs
        return df
    
    def WCSS(self, X_sorted):
        centroid_idxs = self.predict(X_sorted)
        distance = 0
        for i in range(len(X_sorted)):
            x = X_sorted.iloc[i].to_numpy()
            idx = centroid_idxs[i]
            distance += np.power(np.linalg.norm(x-self.centroids[idx]),2)
        return distance
        
#%%

MS, SU, SP, RE = get_scores()
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)
scores = get_score_df()
scores_valid, valid_indices = keep_valid()

#%%
XXX = ESG_KMeans()
rk = XXX.get_ranks(scores_valid)
srtd = XXX.sort_by_avg_rank(rk)
XXX.fit(srtd)

final = XXX.predict_assign(srtd)

#%%
range_clusters = list(range(2,130,20))
SSS = []
for n in range_clusters:
    KM = ESG_KMeans(n_clusters = n)
    ranks = KM.get_ranks(scores_valid)
    rsorted = KM.sort_by_avg_rank(rk)
    KM.fit(rsorted)
    SSS.append(KM.WCSS(rsorted)) 

#%%
plt.plot(range_clusters,SSS)
plt.title("Within clusters sum of squares depending on the number of clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
plt.savefig("Figures/WCSS.png", bbox_inches = 'tight')
    
#%% Plot clusters
p = sns.pairplot(final, hue = 'cluster', x_vars=['MS', 'SU','SP','RE'],
    y_vars=['MS','SU','SP','RE'])
p.fig.suptitle('Clusters for pairs of score ranks', y = 1.01)
plt.savefig("Figures/KMeans_ranks.png", bbox_inches = 'tight')

#%% Stats

mean_ranks = final.groupby('cluster').mean()
std_ranks = final.groupby('cluster').std()