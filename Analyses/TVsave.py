# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:12:20 2024

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

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
ESGTV = SM.make_score_2(SMK, n_classes = 7, gaussian = False)
ESGTV2 = GSM.make_score_2(SMG, n_classes = 7, gaussian = True)


# =============================================================================
# #%% Plot 3D
# agencies = ['MS', 'SU', 'SP', 'RE']
# 
# # Generate triplets of agencies
# triplets = [(agencies[i], agencies[j], agencies[k]) 
#             for i in range(len(agencies)) 
#             for j in range(i + 1, len(agencies)) 
#             for k in range(j + 1, len(agencies))]
# 
# Gmodel = GSM.model
# weights_, means_, covariances_ = Gmodel.weights_, Gmodel.means_, Gmodel.covariances_
# 
# # Function to ensure the covariance matrix is positive definite
# def make_positive_definite(cov):
#     try:
#         # Try Cholesky decomposition to test positive definiteness
#         np.linalg.cholesky(cov)
#     except np.linalg.LinAlgError:
#         # Add a small value to the diagonal elements if not positive definite
#         cov += np.eye(cov.shape[0]) * 1e-6
#     return cov
# 
# fig = plt.figure(figsize=(20, 10))
# num_triplets = len(triplets)
# 
# for idx, triplet in enumerate(triplets):
#     ax = fig.add_subplot(1, num_triplets, idx + 1, projection='3d')
#     ax.set_title(f'Triplet: {triplet}')
#     
#     # Indices of the triplet coordinates
#     triplet_indices = [agencies.index(agency) for agency in triplet]
#     
#     x, y = np.mgrid[0:333:1, 0:333:1]
#     pos = np.dstack((x, y))
#     
#     for i, (mean, covar, weight) in enumerate(zip(means_, covariances_, weights_)):
#         # Extract 2D means and covariances for the first two dimensions of the triplet
#         mean_2d = mean[triplet_indices[:2]]
#         covar_2d = covar[np.ix_(triplet_indices[:2], triplet_indices[:2])]
#         #covar_2d = make_positive_definite(covar_2d)  # Ensure positive definiteness
#         
#         rv = multivariate_normal(mean_2d, covar_2d, allow_singular = True)
#         density = rv.pdf(pos)
#         
#         # Plot 3D density surface
#         ax.plot_surface(x, y, density, cmap='viridis', alpha=0.7, rstride=1, cstride=1, linewidth=0, antialiased=False)
#         
#         # Use color to represent the value of the third dimension
#         third_dim_value = mean[triplet_indices[2]]
#         ax.scatter(mean_2d[0], mean_2d[1], 0, c=third_dim_value, cmap='viridis', edgecolor='k', s=100)
#         
#     ax.set_xlabel(triplet[0])
#     ax.set_ylabel(triplet[1])
#     ax.set_zlabel('Density')
# 
# fig.colorbar(ax.plot_surface(x, y, density, cmap='viridis', alpha=0.7), ax=ax, orientation='horizontal', label=triplet[2])
# plt.tight_layout()
# plt.show()
# 
# =============================================================================

#%% Plot 3D
agencies = ['MS', 'SU', 'SP', 'RE']

# Generate triplets of agencies
triplets = [(agencies[i], agencies[j], agencies[k]) 
            for i in range(len(agencies)) 
            for j in range(i + 1, len(agencies)) 
            for k in range(j + 1, len(agencies))]

Gmodel = GSM.model
weights_, means_, covariances_ = Gmodel.weights_, Gmodel.means_, Gmodel.covariances_

# Function to ensure the covariance matrix is positive definite
def make_positive_definite(cov):
    try:
        # Try Cholesky decomposition to test positive definiteness
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Add a small value to the diagonal elements if not positive definite
        cov += np.eye(cov.shape[0]) * 1e-6
    return cov

# Number of samples to generate
num_samples = 10000

# Plotting
fig = plt.figure(figsize=(20, 10))
num_triplets = len(triplets)

for idx, triplet in enumerate(triplets):
    ax = fig.add_subplot(1, num_triplets, idx + 1, projection='3d')
    ax.set_title(f'Triplet: {triplet}')
    
    # Indices of the triplet coordinates
    triplet_indices = [agencies.index(agency) for agency in triplet]
    
    all_samples = []
    for i, (mean, covar, weight) in enumerate(zip(means_, covariances_, weights_)):
        # Extract 3D means and covariances for the current triplet
        mean_3d = mean[triplet_indices]
        covar_3d = covar[np.ix_(triplet_indices, triplet_indices)]
        #covar_3d = make_positive_definite(covar_3d)  # Ensure positive definiteness
        
        # Generate samples
        samples = np.random.multivariate_normal(mean_3d, covar_3d, int(num_samples * weight))
        all_samples.append(samples)
        
        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], label=f'Component {i+1}')
        
    all_samples = np.vstack(all_samples)
    
    ax.set_xlabel(triplet[0])
    min_rank, max_rank = 0,332
    ax.set_xlim(min_rank, max_rank)
    ax.set_ylim(min_rank, max_rank)
    ax.set_zlim(min_rank, max_rank)
    ax.set_ylabel(triplet[1])
    ax.set_zlabel(triplet[2])
    ax.legend()

plt.tight_layout()
plt.show()

#%% Kendall tau for clusters

tauC = 0
tauG = 0
for agency in dict_agencies:
    tauC += kendalltau(scores_ranks[agency][valid_indices], SM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)

    tauG += kendalltau(scores_ranks[agency], GSM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    
#%% Rank stats

mn, st = SM.get_mean_ranks(), SM.get_std_ranks()
mng, stg = GSM.get_mean_ranks(), GSM.get_std_ranks()

#SM.save_model('kmeans.pkl')
#GSM.save_model('gauss.pkl')

load_SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_SM.load_model('kmeans.pkl')
k_scores = load_SM.get_predictions()
ESGTV3 = load_SM.make_score_2(k_scores, n_classes = 7, gaussian = False)

load_GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_GSM.load_model('gauss.pkl')
full_scores = load_GSM.get_predictions()
ESGTV4 = load_GSM.make_score_2(full_scores, n_classes = 7, gaussian = True)

tauLC = 0
tauLG = 0
for agency in dict_agencies:
    tauLC += kendalltau(scores_ranks[agency][valid_indices], load_SM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)

    tauLG += kendalltau(scores_ranks[agency], load_GSM.full_ranks['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    
#%% Plot classes

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
plt.figure()
sns.histplot(data = load_SM.full_ranks, x = 'sorted_labels', hue = 'sorted_labels', palette = cmap, legend = False)

plt.figure()
s = sns.pairplot(load_SM.full_ranks[['MS', 'SU','SP','RE','sorted_labels']], hue = 'sorted_labels', corner = True)
#s.fig.suptitle('Classes obtained with a Gaussian Mixture Model', y = 1.03)
s.fig.suptitle('Classes obtenues avec un GMM', y = 1.03)
s._legend.set_title('Numéro du cluster')
plt.savefig("Figures/gmm_classes.png", bbox_inches = 'tight')

plt.figure()
ks = sns.pairplot(load_GSM.full_ranks[['MS', 'SU','SP','RE','sorted_labels']], hue = 'sorted_labels', corner = True)
#ks.fig.suptitle('Classes obtained with a K-Means model', y = 1.03)
ks.fig.suptitle('Classes obtenues avec un modèle K-Means', y = 1.03)
ks._legend.set_title('Numéro du cluster')
plt.savefig("Figures/k_clusters.png", bbox_inches = 'tight')

#%% Average rank
avg_rank = pd.DataFrame({'AVG':scores_ranks.mean(axis = 1)})

SMA = ScoreMaker(avg_rank, dict_agencies, valid_tickers, valid_indices, 7)
GSMA = ScoreMaker(avg_rank, dict_agencies, valid_tickers, valid_indices, 7)

SMKA = SMA.kmeans()
SMGA, taumaxA = GSMA.gmm_1d()

# ESG Target variables
ESGTVA = SMA.make_score_2(SMKA, n_classes = 7, gaussian = False)
ESGTV2A = GSM.make_score_2(SMGA, n_classes = 7, gaussian = True)

#tauCA = kendalltau(avg_rank['AVG'], SMKA['sorted_labels'], variant = 'c').statistic
#tauGA = kendalltau(avg_rank['AVG'], SMGA['sorted_labels'], variant = 'c').statistic
tauCA, tauGA = 0,0
for agency in dict_agencies:
    tauCA += kendalltau(scores_ranks[agency][valid_indices], SMKA['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    tauGA += kendalltau(scores_ranks[agency][valid_indices], SMGA['sorted_labels'], variant = 'c').statistic / len(dict_agencies)


plt.figure()
un = sns.histplot(data = SMGA, x='AVG', binwidth = 5, hue = 'sorted_labels')
plt.title('Classes obtained with a 1-D Gaussian Mixture Model \n Kendall Tau : {tauG}'.format(tauG = round(tauGA,2)), y = 1.03)
#plt.savefig("Figures/gmm1d.png", bbox_inches = 'tight')

plt.figure()
kun = sns.histplot(data = SMKA, x='AVG', binwidth = 5, hue = 'sorted_labels')
plt.title('Classes obtained with a 1-D K-Means model, \n Kendall Tau : {tauK}'.format(tauK = round(tauCA,2)), y = 1.03)
#plt.savefig("Figures/kmeans1d.png", bbox_inches = 'tight')

#%%
taus = np.zeros((7,3))

for i in range(1,8):
    
    SMB = ScoreMaker(avg_rank, dict_agencies, valid_tickers, valid_indices, 7)
    GSMB = ScoreMaker(avg_rank, dict_agencies, valid_tickers, valid_indices, 7)
    
    SMKB = SMB.kmeans()
    SMGB, taumaxB = GSMB.gmm_1d()
    
    # ESG Target variables
    ESGTVB = SMA.make_score_2(SMKB, n_classes = i, gaussian = False)
    ESGTV2B = GSM.make_score_2(SMGB, n_classes = i, gaussian = True)
    
    #tauCB = kendalltau(avg_rank['AVG'], SMKB['sorted_labels'], variant = 'c').statistic
    #tauGB = kendalltau(avg_rank['AVG'], SMGB['sorted_labels'], variant = 'c').statistic
    tauCB, tauGB = 0,0
    for agency in dict_agencies:
        tauCB += kendalltau(scores_ranks[agency][valid_indices], SMKB['sorted_labels'], variant = 'c').statistic / len(dict_agencies)

        tauGB += kendalltau(scores_ranks[agency][valid_indices], SMGB['sorted_labels'], variant = 'c').statistic / len(dict_agencies)
    taus[i-1] = [i,round(tauCB,3), round(tauGB,3)]
    
    
plt.figure()
plt.plot(range(1,8),taus[:,1], 'o-', label = 'K-Means')
plt.plot(range(1,8),taus[:,2], 'o-', label = 'GMM')
plt.xlabel('Number of classes')
plt.ylabel('Kendall taus')
plt.title('Kendall tau and number of clusters')
plt.legend()
#plt.savefig("Figures/tausclasses.png", bbox_inches = 'tight')