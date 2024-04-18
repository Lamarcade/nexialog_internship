# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:30:00 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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

#%% Kmeans

range_n_clusters = [3, 4, 5, 6, 7,8, 9, 10]
iners = []

for n_clusters in range_n_clusters:
    plt.plot(figsize = (30,20))
    
    ax = plt.gca()
    # The 1st subplot is the silhouette plot

    ax.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(scores_ranks) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scores_ranks)
    iners.append(clusterer.inertia_)
    print("Inertia : ", str(clusterer.inertia_))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(scores_ranks, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scores_ranks, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()


#%%
n_clusters = 10
clusterer = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = clusterer.fit_predict(scores_ranks)
cluster_centers = clusterer.cluster_centers_
dists = 0

for i in range(len(scores_ranks)):
    x = scores_ranks.iloc[i].to_numpy()
    dists += np.linalg.norm(x-cluster_centers[cluster_labels[i]])**2
    
#%%
x =  scores_ranks.iloc[0].to_numpy()
cc = cluster_centers[cluster_labels[0]]
dd = np.linalg.norm(x-cluster_centers[cluster_labels[0]])

#%%
n_range_clusters = list(range(2,130,20))
IS = []

for n_clusters in n_range_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scores_ranks)
    IS.append(clusterer.inertia_)

#%%
plt.plot(n_range_clusters,IS)
plt.title("Inertia depending on the number of clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()