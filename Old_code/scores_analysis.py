# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:37:31 2024

@author: LoïcMARCADET
"""

#%% Data import
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt  
from matplotlib_venn import venn3
import numpy as np
import venn

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

#print(MS.head())
#print(SP.head())
#print(SU.head())

# Modify RICs to get the same format
RE['Constituent RIC'] = RE['Constituent RIC'].str.extract(r'^([^\.]+)')
RE.drop_duplicates(subset = ['Constituent RIC'], inplace=True, ignore_index = True)

# Check if symbols are the same 
print("MSCI, SP, SU, RE count:",len(MS),len(SP),len(SU), len(RE))
print("Equal symbols between MSCI and SP:", (SP['Tag'] == MS['Colonne2']).sum())
print("Equal symbols between SU and SP:", (SP['Tag'] == SU['Symbol']).sum())

#print("Equal symbols between RE and SP:", (RE['Constituent RIC'] == SU['Symbol']).sum())

SPs = SP['Tag']
REs = RE['Constituent RIC']
common_values = pd.merge(SPs, REs, how='inner', left_on= 'Tag', right_on = 'Constituent RIC') 

new_RE = RE[RE['Constituent RIC'].isin(common_values['Tag'])]
print("Equal symbols between RE and SP:",len(new_RE))

#%% RE merge

# Add NAN rows for the missing tickers in Refinitiv
REm = pd.merge(new_RE, SP, how='outer', left_on='Constituent RIC', right_on = 'Tag')

RE_order = {'A+': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D+': 9}

RE_ordered = REm.copy()
RE_ordered['Order'] = RE_ordered["ESG_Grade_01/01/2024"].map(RE_order)
RE_ordered = RE_ordered.sort_values(by='Order')

#%% Data visualisation

#%% Scores distribution
MScol = "Colonne1"
SPcol = "Score"
SUcol = "Score"
REcol = "ESG_Grade_01/01/2024"

# Order MSCI grades
MSsort = MS.copy()
order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
MSsort[MScol] = pd.Categorical(MSsort[MScol], categories=order, ordered=True)
MSsort = MSsort.sort_values(by=MScol)

REsort = REm.copy()
RE_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-']
REsort[REcol] = pd.Categorical(REsort[REcol], categories=RE_order, ordered=True)
REsort = REsort.sort_values(by=REcol)
  
cmap = 'GnBu_d'
sns.set_theme(style="darkgrid")
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (10, 16), constrained_layout=True)

sns.histplot(data=MSsort, x= MScol, hue = MScol, palette = cmap, ax = ax1, multiple ='stack', legend = False)

sns.histplot(data=REsort, x= REcol, hue = REcol, palette = cmap, ax = ax2, multiple ='stack', legend = False)

sns.histplot(data=SP, x= SPcol, hue = SPcol, palette = cmap, ax = ax3, multiple ='stack', legend = False)

sns.histplot(data=SU, x= SUcol, discrete=True, hue = SUcol, palette = cmap, ax = ax4, multiple ='stack', legend = False)

ax1.set_title('MSCI Scores distribution')
ax2.set_title('Refinitiv Scores distribution')
ax3.set_title('S&P Scores distribution')
ax4.set_title('Sustainalytics Scores distribution')
plt.savefig("Figures/distributions.png")

#%% S&P Sectors

# =============================================================================
# sectors_count = SP['Sector'].value_counts()
# plt.figure()
# sns.barplot(data = sectors_count, orient='h', legend = False)
# plt.title("Number of companies in each sector for S&P data")
# plt.show()
# =============================================================================
SP_red = SP.copy()
SP_red = SP_red[['Tag', 'Industry', 'Score']]

sectors_count = SP_red['Industry'].value_counts()

# Create a CategoricalDtype with the correct order
sorted_sectors = pd.CategoricalDtype(categories=sectors_count.index, ordered=True)

# Sort the DataFrame based on the custom order of 'Industry'
sorted_SP = SP_red.sort_values(by='Industry')
sorted_SP['Sector'] = sorted_SP['Industry'].astype(sorted_sectors)

# Sort the DataFrame based on 'Sector' and 'Score'
sorted_SP = sorted_SP.sort_values(by=['Sector', 'Score'], ascending=[True, False])
sorted_SP['Acronym'] = sorted_SP['Sector'].str.extract(r'([A-Z]{3})')

plt.figure()
sns.histplot(data = sorted_SP, y = 'Acronym', hue = 'Industry', legend = False)
plt.title("Number of companies in each sector for S&P data")
plt.savefig("Figures/sectors_count.png")

plt.figure()
sns.boxplot(data = sorted_SP, y ="Acronym", x=SPcol, hue = 'Sector', palette = cmap, legend = False)
plt.title("Scores in each sector for S&P data")
plt.savefig("Figures/sectors_scores.png")

#%% Refinitiv sectors 
RE_red = RE_ordered.copy()
RE_red = RE_red[['Tag', 'NCAIS', 'ESG_Score_01/01/2024']]
# Round to nearest integer in Refinitiv scores
RE_red['Score'] = RE_red['ESG_Score_01/01/2024'].str.replace(',', '.').astype(float).round()

nc_count = RE_red['NCAIS'].value_counts()

# Create a CategoricalDtype with the correct order
sorted_nc = pd.CategoricalDtype(categories=nc_count.index, ordered=True)

# Sort the DataFrame based on the custom order of 'NCAIS'
sorted_RE = RE_red.sort_values(by='NCAIS')
sorted_RE['Sector'] = sorted_RE['NCAIS'].astype(sorted_nc)

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
# Sort the DataFrame based on 'NCAIS' and score
sorted_RE = sorted_RE.sort_values(by=['Sector', 'Score'], ascending=[True, False])
sorted_RE['Acronym'] = sorted_RE['Sector'].map(NCAIS_map)

plt.figure()
sns.histplot(data = sorted_RE, y = 'Acronym', hue = 'Acronym', legend = False)
plt.title("Number of companies in each sector for Refinitiv data")
plt.savefig("Figures/RE_sectors.png", bbox_inches="tight")

plt.figure()
sns.boxplot(data = sorted_RE, y ="Acronym", x = 'Score', hue = 'Acronym', palette = cmap, legend = False)
plt.title("Scores in each sector for Refinitiv data")
plt.savefig("Figures/RE_sectors_scores.png", bbox_inches="tight")

#%% NaN Values

MSnan = MS[MScol].isna()
REnan = REm[REcol].isna()
SPnan = SP[SPcol].isna()
SUnan = SU[SUcol].isna()
MSnanloc = np.where(MSnan == True)[0]
REnanloc = np.where(REnan == True)[0]
SPnanloc = np.where(SPnan == True)[0]
SUnanloc = np.where(SUnan == True)[0]

labels = ['Valid values', 'Missing values']
fig2, (ax5, ax6, ax7, ax8) = plt.subplots(4,1, figsize = (4, 6))

ax5.pie([len(MS[MScol]), MSnan.sum()], autopct='%1.1f%%', labels = labels, colors = ['#63f7b4', '#db5856'], textprops={'size': 'x-small'}, pctdistance=0.6)
ax6.pie([len(RE[REcol]), REnan.sum()], autopct='%1.1f%%', labels = labels, colors = ['#63f7b4', '#db5856'], textprops={'size': 'x-small'}, pctdistance=0.6)
ax7.pie([len(SP[SPcol]), SPnan.sum()], autopct='%1.1f%%', labels = labels, colors = ['#63f7b4', '#db5856'], textprops={'size': 'x-small'}, pctdistance=0.6)
ax8.pie([len(SU[SUcol]), SUnan.sum()], autopct='%1.1f%%', labels = labels, colors = ['#63f7b4', '#db5856'], textprops={'size': 'x-small'}, pctdistance=0.6)

ax5.set_title('MSCI missing values percentage')
ax6.set_title('Refinitiv missing values percentage')
ax7.set_title('S&P missing values percentage')
ax8.set_title('Sustainalytics missing values percentage')

sns.set_theme(style="darkgrid")
plt.savefig("Figures/missing_values.png")

#%% Venn diagram of common missing values
MSSPnan = np.intersect1d(MSnanloc, SPnanloc)
MSSUnan = np.intersect1d(MSnanloc, SUnanloc)
SUSPnan = np.intersect1d(SUnanloc, SPnanloc)

allnan = np.intersect1d(MSSPnan,SUnanloc)

MSunique = len(MSnanloc) - len(MSSPnan) - len(MSSUnan) + len(allnan)
SPunique = len(SPnanloc) - len(MSSPnan) - len(SUSPnan) + len(allnan)
SUunique = len(SUnanloc) - len(SUSPnan) - len(MSSUnan) + len(allnan)

plt.figure()
venn3(subsets = (MSunique, SPunique, len(MSSPnan) - len(allnan), SUunique, len(MSSUnan) - len(allnan),len(SUSPnan) - len(allnan),len(allnan)), set_labels=('MSCI', 'S&P', 'Sustainalytics'))
plt.title('Number of missing values for each agency including common missing values')
plt.savefig("Figures/common_missing_values.png")

#%% 
MSSPnan = np.intersect1d(MSnanloc, SPnanloc)
MSSUnan = np.intersect1d(MSnanloc, SUnanloc)
SUSPnan = np.intersect1d(SUnanloc, SPnanloc)

# No NAN in RE, but missing tickers
tickers_absent_RE = RE[~RE['Constituent RIC'].isin(common_values['Tag'])].index.tolist()
MSREnan = np.intersect1d(MSnanloc, tickers_absent_RE)
SUREnan = np.intersect1d(SUnanloc, tickers_absent_RE)
RESPnan = np.intersect1d(REnanloc, tickers_absent_RE)

MSSPSUnan = np.intersect1d(MSSPnan,SUnanloc)
RESPSUnan = np.intersect1d(RESPnan,SUnanloc)
allnan = np.intersect1d(MSSPSUnan, RESPSUnan)

vlabels = venn.get_labels([set(MSnanloc), set(tickers_absent_RE), set(SUnanloc), set(SPnanloc)], fill=['number'])

plt.figure()
figv, axv = venn.venn4(vlabels, names=['MS', 'RE', 'SU', 'SP'])
plt.title('Number of missing values for each agency including common missing values')
plt.savefig("Figures/common_missing_values_full.png")

#%%
MSvalid = np.where(MSnan == False)[0]
REvalid = np.where(REnan == False)[0]
SUvalid = np.where(SUnan == False)[0]
SPvalid = np.where(SPnan == False)[0]

MSREv = np.intersect1d(MSvalid, REvalid)
MSSUv = np.intersect1d(MSvalid, SUvalid)
MSSPv = np.intersect1d(MSvalid, SPvalid)
RESUv = np.intersect1d(REvalid, SUvalid)
RESPv = np.intersect1d(REvalid, SPvalid)
SUSPv = np.intersect1d(SUvalid, SPvalid)

RESUSPv = np.intersect1d(RESUv,SPvalid)
MSSUSPv = np.intersect1d(MSSUv,SPvalid)
MSRESPv = np.intersect1d(MSREv,SPvalid)
MSRESUv = np.intersect1d(MSREv,SUvalid)
allv = np.intersect1d(MSRESUv,SPvalid)

full_labels = {'0001': len(SPvalid),
 '0010': len(SUvalid),
 '0011': len(SUSPv),
 '0100': len(REvalid),
 '0101': len(RESPv),
 '0110': len(RESUv),
 '0111': len(RESUSPv),
 '1000': len(MSvalid),
 '1001': len(MSSPv),
 '1010': len(MSSUv),
 '1011': len(MSSUSPv),
 '1100': len(MSREv),
 '1101': len(MSRESPv),
 '1110': len(MSRESUv),
 '1111': len(allv)}

plt.figure()
figv, axv = venn.venn4(full_labels, names=['MS', 'RE', 'SU', 'SP'])
plt.title('Common valid values between agencies')
plt.savefig("Figures/valid_values.png")

#%% Scale harmonizing

#minSP, maxSP = SP[SPcol].min(), SP[SPcol].max()
#minSU, maxSU = SU[SUcol].min(), SU[SUcol].max()

SPc = SP.copy()
SUc = SU.copy()
REc = RE_ordered.copy()

# A higher score in SP is better, while it is worse for Sustainalytics
SPc = SPc.dropna(subset=SPcol).sort_values(by = SPcol, ascending = False)
SUc = SUc.dropna(subset=SUcol).sort_values(by = SUcol, ascending = True)

print(SPc[SPcol])
print(SUc[SUcol])
print(REc[REcol])

### Equal intervals method
EIbool = False

# Discretize the column into custom classes
if EIbool:
   SPc[SPcol] = pd.qcut(SPc[SPcol], 7, labels = order)
   SUc[SUcol] = pd.qcut(SUc[SUcol], 7, labels = order) 
   REc['Order'] = pd.qcut(REc[REcol], 7, labels = order) 
else:   
    ### Percentile method 
    classes_perc = MSsort[MScol].value_counts().reindex(order) / (503- len(MSnanloc))

    # Calculate custom bin edges based on percentiles
    percentiles = np.insert(np.cumsum(classes_perc.values),0,0)
    SPc[SPcol] = pd.qcut(SPc[SPcol], percentiles, labels = order)
    SUc[SUcol] = pd.qcut(SUc[SUcol], percentiles, labels = order)
    REc['Order'] = pd.qcut(REc['Order'], percentiles, labels = order)


figd, (ax1d, ax2d, ax3d, ax4d) = plt.subplots(4,1, figsize = (10, 16), constrained_layout=True)

sns.histplot(data=MSsort, x= MScol, hue = MScol, palette = cmap, ax = ax1d, multiple ='stack',legend = False)

sns.histplot(data=REc, x= 'Order', hue = REcol, palette = cmap, ax = ax2d, multiple ='stack',legend = False)

sns.histplot(data=SPc, x= SPcol, hue = SPcol, palette = cmap, ax = ax3d, multiple ='stack',legend = False)

sns.histplot(data=SUc, x= SUcol, discrete=True, hue = SUcol, palette = cmap, ax = ax4d, multiple ='stack',legend = False)

ax1d.set_title('MSCI Scores distribution')
ax2d.set_title('Refinitiv Scores distribution')
ax3d.set_title('S&P Scores distribution')
ax4d.set_title('Sustainalytics Scores distribution')
plt.savefig("Figures/new_distributions.png")

# Sort again by index to compare companies
SPc = SPc.sort_index()
SUc = SUc.sort_index()
REc = REc.sort_index()

#%% Class Analysis

# S&P and Sustainalytics
# Confusion matrix using pd.crosstab
SPSU_matrix = pd.crosstab(SPc[SPcol], SUc[SUcol], rownames=['SP'], colnames=['Su'], dropna=False)

# Reorder the confusion matrix based on the class order
SPSU_matrix = SPSU_matrix.reindex(index=order, columns=order, fill_value=0)


# MSCI and SP
SPMS_matrix = pd.crosstab(SPc[SPcol], MS[MScol], rownames=['SP'], colnames=['MSCI'], dropna=False)
SPMS_matrix = SPMS_matrix.reindex(index=order, columns=order, fill_value=0)

# MSCI and Sustainalytics
MSSU_matrix = pd.crosstab(MS[MScol], SUc[SUcol], rownames=['MSCI'], colnames=['SU'], dropna=False)
MSSU_matrix = MSSU_matrix.reindex(index=order, columns=order, fill_value=0)

# Refinitiv and MSCI
REMS_matrix = pd.crosstab(REc['Order'], MS[MScol], rownames=['RE'], colnames=['MSCI'], dropna=False)
REMS_matrix = REMS_matrix.reindex(index=order, columns=order, fill_value=0)

# Refinitiv and SP
RESP_matrix = pd.crosstab(REc['Order'], SPc[SPcol], rownames=['RE'], colnames=['SP'], dropna=False)
RESP_matrix = RESP_matrix.reindex(index=order, columns=order, fill_value=0)

# Refinitiv and Sustainalytics
RESU_matrix = pd.crosstab(REc['Order'], SUc[SUcol], rownames=['RE'], colnames=['SU'], dropna=False)
RESU_matrix = RESU_matrix.reindex(index=order, columns=order, fill_value=0)

# Plot confusion matrices
mfig, axes = plt.subplots(1, 3)

sns.heatmap(SPSU_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes[0])
sns.heatmap(SPMS_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes[1])
sns.heatmap(MSSU_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes[2])

axes[0].set_xlabel('SP')
axes[0].set_ylabel('SU')
axes[1].set_xlabel('SP')
axes[1].set_ylabel('MSCI')
axes[2].set_xlabel('MSCI')
axes[2].set_ylabel('SU')


mfig.suptitle('Confusion matrices between the agencies')
mfig.tight_layout()
if EIbool:
    plt.savefig("Figures/confusions_EI.png")
else:
    plt.savefig("Figures/confusions.png")
    
# Plot confusion matrices refinitiv
m2fig, axes2 = plt.subplots(1, 3)

sns.heatmap(REMS_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes2[0])
sns.heatmap(RESP_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes2[1])
sns.heatmap(RESU_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax = axes2[2])

axes2[0].set_xlabel('MSCI')
axes2[0].set_ylabel('RE')
axes2[1].set_xlabel('SP')
axes2[1].set_ylabel('RE')
axes2[2].set_xlabel('SU')
axes2[2].set_ylabel('RE')


m2fig.suptitle('Confusion matrices between the agencies')
m2fig.tight_layout()
if EIbool:
    plt.savefig("Figures/confusions_EI_r.png")
else:
    plt.savefig("Figures/confusions_r.png")
    
    
