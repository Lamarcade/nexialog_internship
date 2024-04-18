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

#%% Filter out common Nan values 

MSnan = MS2['Score'].isna()
REnan = RES['Score'].isna()
SPnan = SPS['Score'].isna()
SUnan = SU2['Score'].isna()
MSvalid = np.where(MSnan == False)[0]
REvalid = np.where(REnan == False)[0]
SPvalid = np.where(SPnan == False)[0]
SUvalid = np.where(SUnan == False)[0]

MSSPvalid = np.intersect1d(MSvalid, SPvalid)
MSSUvalid = np.intersect1d(MSvalid, SUvalid)
MSREvalid = np.intersect1d(MSvalid, REvalid)
SUSPvalid = np.intersect1d(SUvalid, SPvalid)
SUREvalid = np.intersect1d(SUvalid, REvalid)
SPREvalid = np.intersect1d(SPvalid, REvalid)

RES['Score'] = RES['Score'].str.replace(',', '.').astype(float).round()

#%% Kendall's tau-c

tauMSSP = kendalltau(MS2['Score'][MSSPvalid], SPS['Score'][MSSPvalid], variant = 'c')
tauMSSU = kendalltau(MS2['Score'][MSSUvalid], SU2['Score'][MSSUvalid], variant = 'c')
tauMSRE = kendalltau(MS2['Score'][MSREvalid], RES['Score'][MSREvalid], variant = 'c')
tauSUSP = kendalltau(SU2['Score'][SUSPvalid], SPS['Score'][SUSPvalid], variant = 'c')
tauSURE = kendalltau(SU2['Score'][SUREvalid], RES['Score'][SUREvalid], variant = 'c')
tauSPRE = kendalltau(SPS['Score'][SPREvalid], RES['Score'][SPREvalid], variant = 'c')


#%% Figs 

fig, axes = plt.subplots(1, 3)

sns.scatterplot(x = MS2['Score'][MSSPvalid], y= SPS['Score'][MSSPvalid], ax = axes[0])
sns.scatterplot(x = MS2['Score'][MSSUvalid], y= SU2['Score'][MSSUvalid], ax = axes[1])
sns.scatterplot(x = MS2['Score'][MSREvalid], y= RES['Score'][MSREvalid], ax = axes[2])

fig2, axes2 = plt.subplots(1, 3)

sns.scatterplot(x = SU2['Score'][SUSPvalid], y= SPS['Score'][SUSPvalid], ax = axes2[0])
sns.scatterplot(x = SU2['Score'][SUREvalid], y= RES['Score'][SUREvalid], ax = axes2[1])
sns.scatterplot(x = SPS['Score'][SPREvalid], y= RES['Score'][SPREvalid], ax = axes2[2])

#%% On same fig
scores = pd.DataFrame({'MSCI':  MS2['Score'], 'SP' : SPS['Score'], 'RE':RES['Score'], 'SU': SU2['Score']})

sns.pairplot(scores, corner = True)