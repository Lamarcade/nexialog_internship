# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:08:25 2024

@author: LoïcMARCADET
"""

#%% Data import
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_scores(path = 'Scores/'):
    
    # Files containing the scores
    msci = 'MSCI scores.csv'
    sust = 'scores_sust_colnames.csv'
    spgl = 'scores_SP_clean.csv'
    refi = 'Refinitiv_SP500_ESG_score_extract.csv'

    # Create dataframes
    MS = pd.read_csv(path+msci, sep = ';')
    SU = pd.read_csv(path+sust, sep = " ")
    SP = pd.read_csv(path+spgl, sep = ';')
    RE = pd.read_csv(path+refi, sep = ';')

    # Modify Refinitiv RICs to get the same format
    RE['Constituent RIC'] = RE['Constituent RIC'].str.extract(r'^([^\.]+)')
    RE.drop_duplicates(subset = ['Constituent RIC'], inplace=True, ignore_index = True)
    
    # Complete missing RICs with NAN in Refinitiv dataset
    SPs = SP['Tag']
    REs = RE['Constituent RIC']
    common_values = pd.merge(SPs, REs, how='inner', left_on= 'Tag', right_on = 'Constituent RIC') 
    
    new_RE = RE[RE['Constituent RIC'].isin(common_values['Tag'])]
    REm = pd.merge(new_RE, SP, how='outer', left_on='Constituent RIC', right_on = 'Tag')
    REm['Constituent RIC'] = SP['Tag'].copy()
    
    # The Refinitiv dataset has a NaN row at index 503
    if (len(REm) == 504):
        REm.drop(REm.index[-1], inplace=True)
    
    return MS, SU, SP, REm

#MS, SU, SP, RE = get_scores()
#%% Retrieve score columns and homogeneise

def reduced_df(MS, SU, SP, RE): 
    MSS = MS[['Colonne2', 'Colonne1']].rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
    SPS = SP[['Tag', 'Score']]
    SUS = SU[['Symbol', 'Score']].rename(columns = {'Symbol':'Tag'})
    RES = RE[['Constituent RIC', 'ESG_Score_01/01/2024']].rename(columns = {'Constituent RIC':'Tag','ESG_Score_01/01/2024':'Score'})
    
    MS_order = {'AAA': 6, 'AA': 5, 'A': 4, 'BBB': 3, 'BB': 2, 'B': 1, 'CCC': 0}
    
    MSS['Score'] = MSS['Score'].map(MS_order)
    
    #A higher score in Sustainalytics is worse
    reverse = lambda x: 100-x
    SUS['Score'] = SUS['Score'].map(reverse)
    
    # Round to nearest integer in Refinitiv scores
    RES['Score'] = RES['Score'].str.replace(',', '.').astype(float).round()
    return(MSS,SUS,SPS,RES)

# Homogeneise but do not reduce
def homogen_df(MS, SU, SP, RE): 
    MSS = MS.copy().rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
    SPS = SP.copy()
    SUS = SU.copy().rename(columns = {'Symbol':'Tag'})
    RES = RE.copy().rename(columns = {'Constituent RIC':'Tag'})
    
    MS_order = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}
    
    MSS['Score'] = MSS['Score'].map(MS_order)
    
    #A higher score in Sustainalytics is worse
    reverse = lambda x: 100-x
    SUS['Score'] = SUS['Score'].map(reverse)
    
    # Round to nearest integer in Refinitiv scores
    RES['Score'] = RES['ESG_Score_01/01/2024'].str.replace(',', '.').astype(float).round()
    return(MSS,SUS,SPS,RES)

def reduced_mixed_df(MS, SU, SP, RE): 
    MSS = MS[['Colonne2', 'Colonne1']].rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
    SPS = SP[['Tag', 'Score']]
    SUS = SU[['Symbol', 'Score']].rename(columns = {'Symbol':'Tag'})
    RES = RE[['Constituent RIC', 'ESG_Score_01/01/2024']].rename(columns = {'Constituent RIC':'Tag','ESG_Score_01/01/2024':'Score'})
    # Round to nearest integer in Refinitiv scores
    RES['Score'] = RES['Score'].str.replace(',', '.').astype(float).round()
    return(MSS,SUS,SPS,RES)

#MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)
#%% Create a dataframe with all the scores

# All the agencies
#dict_agencies = {'MS':  MSS['Score'], 'SU': SUS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score']}

def get_score_df(dic):
    return pd.DataFrame(dic)

#scores = get_score_df()

def keep_valid(score_df):
    scores_valid = score_df[~(score_df.isna().any(axis=1))]
    return (scores_valid, scores_valid.index)

#scores_valid, valid_indices = keep_valid()

def standardise_df(score_df):
    scaler = StandardScaler()
    return(scaler.fit_transform(score_df))

#std_scores = standardise_df()

#%% Gini coefficient for sector neutrality
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def gini(score_means):
    nb_nan = np.isnan(score_means).sum()
    means = [x for x in score_means if not np.isnan(x)]
    
    means_sort = np.sort(np.append(means,0), axis=None)
    n = len(means_sort)
    cumulative_means = np.cumsum(means_sort)
    cumulative_means_std = cumulative_means/cumulative_means[-1]
    
    equality = [k/(n-1) for k in range(n)]
    
    return simpson(2*(equality - cumulative_means_std), x=equality), cumulative_means_std, equality, nb_nan

def plot_gini(gini, cumulative_means_std, equality, nb_nan, agency, method):
    plt.figure()
    plt.plot(equality,equality, color = 'bisque')
    plt.plot(equality, cumulative_means_std, 'o:r', ms = 1)
    plt.fill(equality, cumulative_means_std, hatch = '/', color = 'lightgrey', label = 'Gini area')
    plt.xlabel('Pourcentage cumulé de secteurs sur ' + str(len(equality)) + ' secteurs (' + str(nb_nan) + ' variances invalides')
    plt.ylabel('Pourcentage cumulé de moyenne ESG')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Gini = '+ str(round(gini, 3)) + ' pour la ' + str(method) + ' de scores ' + str(agency))
    plt.legend()
    plt.show()
            
def plot_gini2(nb_plots, ginis, cumus, equas, nb_nans, agencies, methods):
    if nb_plots <= 3:
        return 'Not enough plots'
    elif nb_plots <= 6:
        i, j = 2, nb_plots // 2 + nb_plots % 2
    else:
        return 'Too many plots'
    
    fig, ax = plt.subplots(i, j, figsize=(20, 15))
    
    for idx, (k, l) in enumerate([(x, y) for x in range(i) for y in range(j)][:nb_plots]):
        ax[k][l].plot(equas[idx], equas[idx], color='bisque')
        ax[k][l].plot(equas[idx], cumus[idx], 'o:r', ms=1)
        ax[k][l].fill(equas[idx], cumus[idx], hatch='/', color='lightgrey', label='Gini area')
        ax[k][l].set_xlabel(f'Pourcentage cumulé de secteurs sur {len(equas[idx])} secteurs ({nb_nans[idx]} variances invalides)')
        ax[k][l].set_ylabel('Pourcentage cumulé de moyenne ESG')
        ax[k][l].yaxis.set_major_formatter(PercentFormatter(1))
        ax[k][l].xaxis.set_major_formatter(PercentFormatter(1))
        ax[k][l].set_title(f'Gini = {round(ginis[idx], 3)} pour la {methods[idx]} de scores {agencies[idx]}')
        ax[k][l].legend()

    plt.show()

    