# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:34:20 2024

@author: LoïcMARCADET
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from ScoreGetter import ScoreGetter

from ESG.scores_utils import *

SG = ScoreGetter('ESG/Scores/')
SG.reduced_df()
scores_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()


def maps(scores_ranks, n_classes, valid_tickers, reverse = False, get_all = False):
    ranks = scores_ranks.copy()
    
    # Transforms the ranks into 0 to n_classes-1 scores
    list_scores = list(range(n_classes))
    thresholds = [i* 333/n_classes for i in range(n_classes)]
    def mapping(score):
        for i in range(len(thresholds)):
            if score <= thresholds[i]:
                return(list_scores[i]-1)
        return n_classes-1
    
    # A higher rank indicates a low score
    #reverse_rank = lambda x: 334-x
    
    for agency in ranks.columns:
        #ranks[agency] = ranks[agency].map(reverse_rank)
        ranks[agency] = ranks[agency].map(mapping)
    
    return(ranks)
    
scores_df = maps(scores_ranks, 7, valid_tickers)

# Worst score approach
ESGTV3, all_ranks = SG.worst_score(scores_ranks, n_classes = 7, get_all = True)
ESGTV4 = SG.worst_score(scores_ranks, n_classes = 7, reverse = True)

ESGTV5 = pd.DataFrame({'Tag': ESGTV3['Tag'], 'Score': round(all_ranks.mean(axis = 1)).astype(int)})

range_number = range(len(ESGTV3))
# =============================================================================
# plt.figure(figsize = (20,6))
# plt.plot(range_number, ESGTV3['Score'], 'bo', label = 'Worst')
# plt.plot(range_number, ESGTV4['Score'], 'go', label = 'Best')
# plt.plot(range_number, ESGTV5['Score'], 'ro', label = 'Average')
# plt.legend()
# plt.show()
# =============================================================================
esg_df = pd.DataFrame({'Tag': valid_tickers, 'Worst': ESGTV3['Score'], 'Best': ESGTV4['Score'], 'Mean': ESGTV5['Score']})

dist_df = pd.DataFrame({'Worst': ESGTV3['Score'], 'Best': ESGTV4['Score'], 'Mean': ESGTV5['Score']})

#%%
correls = np.zeros((4,4))

agency_order = ['MS', 'SU', 'SP', 'RE']
names = ['MSCI', 'Sust.', 'S&P', 'Refi.']
for i in range(4):
    correls[i][i] = 1
    for j in range(4):
        if i!= j:
            correls[i][j], _ = kendalltau(scores_df[agency_order[i]],scores_df[agency_order[j]], variant = 'b')

mask = np.triu(np.ones_like(correls, dtype=bool))

#sns.set_theme()
sns.reset_defaults()

plt.figure()
sns.heatmap(correls, annot = True, mask = mask, xticklabels=names, yticklabels = names, fmt = ".2f", cmap = 'viridis')
fr = False
if fr:
    plt.title('Corrélation de rang des scores ESG \n pour les entreprises du S&P 500')
    plt.savefig('Figures/policies_correlations_fr.png')
else:
    plt.title('Rank correlation of discretized ESG scores \n for companies in the S&P 500')
    plt.savefig('Figures/policies_correlations.png')    
    
#%%
avgkendall = [0,0,0]

for i, policy in enumerate(['Worst','Best','Mean']):
    for agency in dict_agencies:
        avgkendall[i] += kendalltau(scores_ranks[agency][valid_indices], esg_df[policy], variant = 'c').statistic / len(dict_agencies)

# %% Retrieve dataframes

MS, SU, SP, RE = get_scores('ESG/Scores/')
MSH, SUH, SPH, REH = homogen_df(MS, SU, SP, RE)
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)

# All the agencies
dict_agencies = {'MS':  MSS['Score'], 'SU': SUS['Score'], 'SP': SPS['Score'],
    'RE': RES['Score']}
scores = get_score_df(dict_agencies)
scores_valid, valid_indices = keep_valid(scores)
std_scores = standardise_df(scores)


agencies_order = ['MS', 'SU', 'SP', 'RE']
n_agencies = len(agencies_order)

# Valid values for each agency
valid_locs = []
for agency in agencies_order:
    valid_locs.append(np.where(scores[agency].isna() == False)[0])

# %% Scores distribution
MSR, SUR, SPR, RER = reduced_mixed_df(MS, SU, SP, RE)

base_dic = {'MS':  MSR['Score'], 'SU': SUR['Score'], 'SP': SPR['Score'],
    'RE': RER['Score']}

base_scores = get_score_df(base_dic)

# %% Scale harmonizing

# Order MSCI grades
MSsort = MS.copy()
order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
pandas_order = ['CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']
MSsort['Colonne1'] = pd.Categorical(
    MSsort['Colonne1'], categories=order, ordered=True)
MSsort = MSsort.sort_values(by='Colonne1')


def harmonize(MSsort=MSsort, SUS=SUS, SPS=SPS, RES=RES, n_classes=7, valid_locs=valid_locs, c_order=pandas_order, EIbool=False):

    SPc, SUc, REc, MSc = SPS.copy(), SUS.copy(), RES.copy(), MSsort.copy()

    SPc = SPc.dropna(subset='Score').sort_values(by='Score', ascending=False)
    SUc = SUc.dropna(subset='Score').sort_values(by='Score', ascending=False)
    REc = REc.dropna(subset='Score').sort_values(by='Score', ascending=False)

    if EIbool:
       SPc['Score'] = pd.qcut(SPc['Score'], n_classes, labels=c_order)
       SUc['Score'] = pd.qcut(SUc['Score'], n_classes, labels=c_order)
       REc['Score'] = pd.qcut(REc['Score'], n_classes, labels=c_order)
    else:
        # Percentile method
        classes_perc = MSsort['Colonne1'].value_counts().reindex(
            c_order) / (len(valid_locs[0]))

        # Calculate custom bin edges based on percentiles
        percentiles = np.insert(np.cumsum(classes_perc.values), 0, 0)

        # SP can have duplicate edges, so we rank the scores with method = first to avoid ties
        try:
            SPc['Score'] = pd.qcut(SPc['Score'], percentiles, labels=c_order)
        except:
            SPc['Score'] = pd.qcut(SPc['Score'].rank(
                method='first'), percentiles, labels=c_order)

        SUc['Score'] = pd.qcut(SUc['Score'], percentiles,
                               labels=c_order, duplicates='drop')
        REc['Score'] = pd.qcut(REc['Score'], percentiles,
                               labels=c_order, duplicates='drop')

    # MSc had to be ordered the other way around for Pandas qcut
    MSc['Colonne1'] = MSc['Colonne1'].cat.reorder_categories(c_order)
    return (MSc, SUc, SPc, REc)


MS_ha, SU_ha, SP_ha, RE_ha = harmonize()
harmonized_dict = {'MS':  MS_ha['Colonne1'], 'SP': SP_ha['Score'],
    'RE': RE_ha['Score'], 'SU': SU_ha['Score']}
scores_ha = get_score_df(harmonized_dict)

# %% Krippendorff

MS_order = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}
    
scores_num = scores_ha.copy()
for agency in scores_num.columns:
    scores_num[agency] = scores_num[agency].map(MS_order)
#%% Kendall-tau
kdf = scores_num.copy()
kdf = kdf[~(kdf.isna().any(axis=1))]

kts = []
for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            ag1, ag2 = agencies_order[i], agencies_order[j]
            kt = kendalltau(scores_valid[ag1], scores_valid[ag2]).statistic
            kts.append((ag1,ag2,kt))

#%%
correls = np.zeros((4,4))

agency_order = ['MS', 'SU', 'SP', 'RE']
names = ['MSCI', 'Sust.', 'S&P', 'Refi.']
for i in range(4):
    correls[i][i] = 1
    for j in range(4):
        if i!= j:
            correls[i][j], _ = kendalltau(kdf[agency_order[i]],kdf[agency_order[j]], variant = 'b')

mask = np.triu(np.ones_like(correls, dtype=bool))

plt.figure()
sns.reset_defaults()
sns.heatmap(correls, annot = True, mask = mask, xticklabels=names, yticklabels = names, fmt = ".2f", cmap = 'viridis')

plt.title('Rank correlation of ESG scores for companies in the S&P 500 \n after converting according to the MSCI quantiles')
plt.savefig('Figures/quant_correlations.png')    

#%%
# Worst score approach
rkdf = kdf.rank()
ESGTV6, q_all_ranks = SG.worst_score(rkdf, n_classes = 7, get_all = True)
ESGTV7 = SG.worst_score(rkdf, n_classes = 7, reverse = True)

ESGTV8 = pd.DataFrame({'Tag': ESGTV3['Tag'], 'Score': round(q_all_ranks.mean(axis = 1)).astype(int)})

range_number = range(len(ESGTV3))
# =============================================================================
# plt.figure(figsize = (20,6))
# plt.plot(range_number, ESGTV3['Score'], 'bo', label = 'Worst')
# plt.plot(range_number, ESGTV4['Score'], 'go', label = 'Best')
# plt.plot(range_number, ESGTV5['Score'], 'ro', label = 'Average')
# plt.legend()
# plt.show()
# =============================================================================
q_esg_df = pd.DataFrame({'Tag': valid_tickers, 'Worst': ESGTV6['Score'], 'Best': ESGTV7['Score'], 'Mean': ESGTV8['Score']})

#%%

q_kendall = [0,0,0]

for i, policy in enumerate(['Worst','Best','Mean']):
    for agency in dict_agencies:
        q_kendall[i] += kendalltau(scores_ranks[agency][valid_indices], q_esg_df[policy], variant = 'c').statistic / len(dict_agencies)
