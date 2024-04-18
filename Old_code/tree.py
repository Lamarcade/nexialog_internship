# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:50:57 2024

@author: LoïcMARCADET
"""

#%% Data import
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import StandardScaler

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

RES['Score'] = RES['Score'].str.replace(',', '.').astype(float).round()

MS_order = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}

MS2 = MSS.copy()
MS2['Score'] = MS2['Score'].map(MS_order)

scores = pd.DataFrame({'MSCI':  MS2['Score'], 'SP' : SPS['Score'], 'RE':RES['Score'], 'SU': SUS['Score']})
scores_nona = scores[~(scores.isna().any(axis=1))]

valid_indices = scores_nona.index

clf = tree.DecisionTreeRegressor(max_depth = 17)

clf = clf.fit(scores_nona[['SP','RE','SU']], scores_nona['MSCI'])
s = clf.score(scores_nona[['SP','RE','SU']], scores_nona['MSCI'])