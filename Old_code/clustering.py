# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:21:32 2024

@author: LoïcMARCADET
"""

#%% Data import
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import gower
import math
from sklearn.cluster import AgglomerativeClustering

plt.close('all') 

path = 'Scores/'
msci = 'MSCI scores.csv'
sust = 'scores_sust_colnames.csv'
spgl = 'scores_SP_clean.csv'
refi = 'Refinitiv_SP500_ESG_score_extract.csv'

MS = pd.read_csv(path+msci, sep =';')
SP = pd.read_csv(path+spgl, sep =';')
SU = pd.read_csv(path+sust, sep = " ")
RE = pd.read_csv(path+refi, sep = ';')

MSc = MS.copy()
SPc = SP.copy()
SUc = SU.copy()

# Modify Refinitiv RICs to get the same format
RE['Constituent RIC'] = RE['Constituent RIC'].str.extract(r'^([^\.]+)')
RE.drop_duplicates(subset = ['Constituent RIC'], inplace=True, ignore_index = True)

SPs = SP['Tag']
REs = RE['Constituent RIC']
common_values = pd.merge(SPs, REs, how='inner', left_on= 'Tag', right_on = 'Constituent RIC') 

# Complete missing RICs with NAN in Refinitiv dataset
new_RE = RE[RE['Constituent RIC'].isin(common_values['Tag'])]
REm = pd.merge(new_RE, SP, how='outer', left_on='Constituent RIC', right_on = 'Tag')
REm['Constituent RIC'] = SP['Tag'].copy()
REc = REm.copy()

#%% Score columns
MSS = MSc[['Colonne2', 'Colonne1']].rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
SPS = SPc[['Tag', 'Score']]
SUS = SUc[['Symbol', 'Score']].rename(columns = {'Symbol':'Tag'})
RES = REc[['Constituent RIC', 'ESG_Score_01/01/2024']].rename(columns = {'Constituent RIC':'Tag','ESG_Score_01/01/2024':'Score'})


#%% Convert MSCI to numerical scores

MS_order = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}

MS2 = MSS.copy()
MS2['Score'] = MS2['Score'].map(MS_order)

#A higher score in Sustainalytics is worse
reverse = lambda x: 100-x

SU2 = SUS.copy()
SU2['Score'] = SU2['Score'].map(reverse)

# Round to nearest integer in Refinitiv scores
RES['Score'] = RES['Score'].str.replace(',', '.').astype(float).round()

#%% Clustering 
scores = pd.DataFrame({'MSCI':  MS2['Score'], 'SP' : SPS['Score'], 'RE':RES['Score'], 'SU': SU2['Score']})
scores_nona = rows_with_nan = scores[~(scores.isna().any(axis=1))]

valid_indices = scores_nona.index

distance_matrix = gower.gower_matrix(scores_nona)

clusters = AgglomerativeClustering(n_clusters = 7, metric = 'precomputed', linkage = 'average')
labels = pd.DataFrame(clusters.fit_predict(distance_matrix), index = valid_indices)

scores_nona['labels'] = labels

#%%

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
plt.plot()
sns.histplot(labels, palette = cmap)



plt.plot()
sns.pairplot(scores_nona, hue = 'labels', corner = True)
