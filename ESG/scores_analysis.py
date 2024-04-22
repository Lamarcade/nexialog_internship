# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:34:03 2024

@author: LoïcMARCADET
"""
# %% Data import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import venn
from scipy.stats import spearmanr

from scores_utils import *

plt.close('all')

# %% Retrieve dataframes

MS, SU, SP, RE = get_scores()
MSH, SUH, SPH, REH = homogen_df(MS, SU, SP, RE)
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)

# All the agencies
dict_agencies = {'MS':  MSS['Score'], 'SU': SUS['Score'], 'SP': SPS['Score'],
    'RE': RES['Score']}
scores = get_score_df(dict_agencies)
scores_valid, valid_indices = keep_valid(scores)
std_scores = standardise_df(scores)


# %% Data visualisation


# %% Scores distribution
MSR, SUR, SPR, RER = reduced_mixed_df(MS, SU, SP, RE)

base_dic = {'MS':  MSR['Score'], 'SU': SUR['Score'], 'SP': SPR['Score'],
    'RE': RER['Score']}

base_scores = get_score_df(base_dic)

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

for (i, agency) in enumerate(base_scores.columns):
    sns.histplot(data=base_scores, x=agency, hue=agency, palette=cmap,
                 ax=ax[i], multiple='stack', legend=False)
    ax[i].set_title(str(agency) + ' scores distribution')

plt.savefig("Figures/base_distributions.png")

# %% Scores distribution after conversions

cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

for (i, agency) in enumerate(scores.columns):
    sns.histplot(data=scores, x=agency, hue=agency, palette=cmap,
                 ax=ax[i], multiple='stack', legend=False)
    ax[i].set_title(str(agency) + ' scores distribution')

plt.savefig("Figures/distributions.png")

# %% Sectors analysis


def sector_analysis(df, sector_col, make_acronym=False, sector_map=None):
    dfc = df.copy()
    sectors_count = dfc[sector_col].value_counts()

    # Create a CategoricalDtype with the correct order
    sorted_sectors = pd.CategoricalDtype(
        categories=sectors_count.index, ordered=True)

    # Create a DataFrame to keep track of the sector count
    sorted_df = dfc.sort_values(by=sector_col)
    sorted_df['Sector_A'] = sorted_df[sector_col].astype(sorted_sectors)

    # Sort the DataFrame based on sector count and score
    sorted_df = sorted_df.sort_values(
        by=['Sector_A', 'Score'], ascending=[True, False])

    # Retrieve 3-letter sector acronym
    if make_acronym:
        sorted_df['Acronym'] = sorted_df[sector_col].str.extract(r'([A-Z]{3})')

    # Map NCAIS sector to its description
    if not (sector_map is None):
        sorted_df['Sector_B'] = sorted_df['Sector_A'].map(sector_map)

    return sorted_df


NCAIS_map = {
'11':	'Agriculture, Forestry, Fishing and Hunting',
'21':	'Mining, Quarrying, and Oil and Gas Extraction',
'22':	'Utilities',
'23':	'Construction',
'31-33':	'Manufacturing',
'41/42':	'Wholesale Trade',
'44-45':	'Retail Trade',
'48-49':	'Transportation and Warehousing',
'51':	'Information',
'52':	'Finance and Insurance',
'53':	'Real Estate and Rental and Leasing',
'54':	'Professional, Scientific, and Technical Services',
'55':	'Management of Companies and Enterprises',
'56':	'Admini. & Support & Waste Management & Remediation Services',
'61':	'Educational Services',
'62':	'Health Care and Social Assistance',
'71':	'Arts, Entertainment, and Recreation',
'72':	'Accommodation and Food Services',
'81':	'Other Services (except Public Administration)',
'91/92':	'Public Administration'
}

SP_sectors = sector_analysis(SPH, 'Industry', make_acronym=True)
RE_sectors = sector_analysis(
    REH, 'NCAIS', make_acronym=False, sector_map=NCAIS_map)

plt.figure()
sns.histplot(data=SP_sectors, y='Acronym', hue='Industry', legend=False)
plt.title("Number of companies in each sector for S&P data")
plt.savefig("Figures/sectors_count.png")

plt.figure()
sns.boxplot(data=SP_sectors, y="Acronym", x='Score',
            hue='Industry', palette=cmap, legend=False)
plt.title("Scores in each sector for S&P data")
plt.savefig("Figures/sectors_scores.png")

plt.figure()
sns.histplot(data=RE_sectors, y='Sector_B', hue='Sector_B', legend=False)
plt.title("Number of companies in each sector for Refinitiv data")
plt.savefig("Figures/RE_sectors.png", bbox_inches="tight")

plt.figure()
sns.boxplot(data=RE_sectors, y="Sector_B", x='Score',
            hue='Sector_B', palette=cmap, legend=False)
plt.title("Scores in each sector for Refinitiv data")
plt.savefig("Figures/RE_sectors_scores.png", bbox_inches="tight")


#%% Gini sector means

RE_means = RE_sectors[['Sector_B','Score']].groupby(['Sector_B']).mean()
SP_means = SP_sectors[['Acronym','Score']].groupby(['Acronym']).mean()

RE_var = RE_sectors[['Sector_B','Score']].groupby(['Sector_B']).var()
SP_var = SP_sectors[['Acronym','Score']].groupby(['Acronym']).var()

all_stats = [RE_means,SP_means,RE_var,SP_var]
ginis, cumus, equas, nb_nans = [],[],[],[]
ag = ['Refinitiv','S&P','Refinitiv','S&P']
methods = ['moyenne', 'moyenne', 'variance', 'variance']
for stat in all_stats:
    stat_gini, stat_cumulative, stat_equality, stat_nan = gini(stat.values)
    ginis.append(stat_gini), 
    cumus.append(stat_cumulative)
    equas.append(stat_equality)
    nb_nans.append(stat_nan)

plot_gini2(4,ginis, cumus, equas, nb_nans, ag, methods)


# %% Valid Values
agencies_order = ['MS', 'SU', 'SP', 'RE']
n_agencies = len(agencies_order)

# Valid values for each agency
valid_locs = []
for agency in agencies_order:
    valid_locs.append(np.where(scores[agency].isna() == False)[0])

# Intersection of valid values as a matrix
valid_inter = [[None] * n_agencies for _ in range(n_agencies)]

for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            valid_inter[i][j] = np.intersect1d(valid_locs[i], valid_locs[j])

# Intersection of valid values for 3 agencies
valid_3 = []

for i in range(n_agencies):
    for j in range(n_agencies):
        for k in range(n_agencies):
            if (j > i) & (k > j):
                valid_3.append(np.intersect1d(
                    valid_inter[i][j], valid_locs[k]))

# Intersection of valid values for all agencies
valid_all = np.intersect1d(valid_3[0], valid_3[1])

full_labels = {'0001': len(valid_locs[2]),  # SP
 '0010': len(valid_locs[1]),  # SU
 '0011': len(valid_inter[1][2]),
 '0100': len(valid_locs[3]),  # RE
 '0101': len(valid_inter[2][3]),
 '0110': len(valid_inter[1][3]),
 '0111': len(valid_3[3]),  # RESUSP
 '1000': len(valid_locs[0]),  # MS
 '1001': len(valid_inter[0][2]),
 '1010': len(valid_inter[0][1]),
 '1011': len(valid_3[0]),  # MSSUSP
 '1100': len(valid_inter[0][3]),
 '1101': len(valid_3[2]),  # MSRESP
 '1110': len(valid_3[1]),  # MSRESU
 '1111': len(valid_all)}

plt.figure()
figv, axv = venn.venn4(full_labels, names=['MS', 'RE', 'SU', 'SP'])
plt.title('Common valid values between agencies')
plt.savefig("Figures/valid_values.png")

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

# %% Plot new distributions
figd, axd = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

for (i, agency) in enumerate(scores_ha.columns):
    sns.histplot(data=scores_ha, x=agency, hue=agency, palette=cmap,
                 ax=axd[i], multiple='stack', legend=False)
    axd[i].set_title(str(agency) + ' scores distribution')

plt.savefig("Figures/new_distributions.png")

# %% Class Analysis

confusions = [[None] * n_agencies for _ in range(n_agencies)]
for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            ag1, ag2 = agencies_order[i], agencies_order[j]
            confusions[i][j] = pd.crosstab(scores_ha[ag1], scores_ha[ag2], rownames=[
                                           ag1], colnames=[ag2], dropna=False)
            confusions[i][j] = confusions[i][j].reindex(
                index=pandas_order, columns=pandas_order, fill_value=0)

# Plot confusion matrices
mfig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=False)

dict_axes = {'01': axes[0][0], '02': axes[0][1], '03': axes[0][2],
                '12': axes[1][0], '13': axes[1][1], '23': axes[1][2]}
for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            sns.heatmap(confusions[i][j], annot=True, fmt='d',
                        cmap='Blues', cbar=False, ax=dict_axes[str(i)+str(j)])
            dict_axes[str(i)+str(j)].set_xlabel(agencies_order[i])
            dict_axes[str(i)+str(j)].set_ylabel(agencies_order[j])

mfig.suptitle('Confusion matrices between the agencies')

plt.savefig("Figures/confusions.png")

# %% Krippendorff

MS_order = {'AAA': 8, 'AA': 7, 'A': 6, 'BBB': 5, 'BB': 4, 'B': 3, 'CCC': 2, 'CC': 1, 'C': 0}
    
scores_num = scores_ha.copy()
for agency in scores_num.columns:
    scores_num[agency] = scores_num[agency].map(MS_order)

scores_kp = []
for key in harmonized_dict.keys():
    scores_kp.append(np.array(scores_num[key]))

def replace_nan(arr):
    if isinstance(arr, np.ndarray):
        if pd.isnull(arr).any():
            arr = np.where(pd.isnull(arr), 'nan', arr)
    elif isinstance(arr, list):
        arr = [replace_nan(x) for x in arr]
    return arr

#result_array = replace_nan(scores_kp)

for i in range(4):
    scores_kp[i] = scores_kp[i].tolist()

#%% Spearman 
df = scores_num.copy()
df = df[~(df.isna().any(axis=1))]

spearmans = []
for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            ag1, ag2 = agencies_order[i], agencies_order[j]
            spear = spearmanr(scores_valid[ag1], scores_valid[ag2]).statistic
            spearmans.append((ag1,ag2,spear))
            
spearmans_conv = []
for i in range(n_agencies):
    for j in range(n_agencies):
        if j > i:
            ag1, ag2 = agencies_order[i], agencies_order[j]
            spear = spearmanr(df[ag1], df[ag2]).statistic
            spearmans_conv.append((ag1,ag2,spear))            


