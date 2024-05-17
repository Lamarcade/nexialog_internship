# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:54:50 2024

@author: LoïcMARCADET
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ScoreGetter:
    def __init__(self, path):
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
        
        # Round to nearest integer in Refinitiv scores
        RE['ESG_Score_01/01/2024'] = RE['ESG_Score_01/01/2024'].str.replace(',', '.').astype(float).round()
        
        # Complete missing RICs with NAN in Refinitiv dataset
        SPs = SP['Tag']
        REs = RE['Constituent RIC']
        common_values = pd.merge(SPs, REs, how='inner', left_on= 'Tag', right_on = 'Constituent RIC') 
        
        new_RE = RE[RE['Constituent RIC'].isin(common_values['Tag'])]
        REm = pd.merge(new_RE, SP, how='outer', left_on='Constituent RIC', right_on = 'Tag')
        REm['Constituent RIC'] = SP['Tag'].copy()
        
        # Drop columns from S&P
        REm = REm.drop(['Name', 'Tag', 'Score'], axis = 1)
        
        # The Refinitiv dataset has a NaN row at index 503
        if (len(REm) == 504):
            REm.drop(REm.index[-1], inplace=True)
        
        self.MS = MS
        self.SU = SU
        self.SP = SP
        self.RE = REm
        
        self.tickerlist = MS['Colonne2']
        
    def set_scores(self, MSS, SUS, SPS, RES):
        self.MSS = MSS
        self.SUS = SUS
        self.SPS = SPS
        self.RES = RES
        
    def get_processed_scores(self):
        return(self.MSS,self.SUS, self.SPS, self.RES)
    
    def name_columns(self):
        MSS = self.MS.copy().rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
        SPS = self.SP.copy()
        SUS = self.SU.copy().rename(columns = {'Symbol':'Tag'})
        RES = self.RE.copy().rename(columns = {'Constituent RIC':'Tag','ESG_Score_01/01/2024':'Score'})
        
        self.set_scores(MSS, SUS, SPS, RES)
        
    def transform_scores(self):
        self.name_columns()
        MS_order = {'AAA': 6, 'AA': 5, 'A': 4, 'BBB': 3, 'BB': 2, 'B': 1, 'CCC': 0}
        
        self.MSS['Score'] = self.MSS['Score'].map(MS_order)
        
        #A higher score in Sustainalytics is worse
        reverse = lambda x: 100-x
        self.SUS['Score'] = self.SUS['Score'].map(reverse)
        
    def reduce(self):
        self.MSS = self.MSS[['Tag', 'Score']]
        self.SPS = self.SPS[['Tag', 'Score']]
        self.SUS = self.SUS[['Tag', 'Score']]
        self.RES = self.RES[['Tag', 'Score']]
    
    def reduced_df(self): 
        self.transform_scores()
        self.reduce()
        
    def homogen_df(self): 
        # Homogeneise but do not reduce
        self.transform_scores()

    def reduced_mixed_df(self): 
        # Reduce but do not transform
        self.name_columns()
        self.reduce()
        
    def make_dict(self):
        self.dict_agencies = {'MS':  self.MSS['Score'], 'SU': self.SUS['Score'], 
                         'SP' : self.SPS['Score'], 'RE': self.RES['Score']}
    
    def get_dict(self):
        return(self.dict_agencies)
    
    def get_valid_indices(self):
        return self.valid_indices
    
    def get_valid_tickers(self):
        return self.valid_tickers
        
    def get_new_df(self):
        self.make_dict()
        return(pd.DataFrame(self.dict_agencies))
    
    def get_score_df(self):
        return self.score_df
    
    def keep_valid(self):
        score_df = self.get_new_df()
        scores_valid = score_df[~(score_df.isna().any(axis=1))]
        self.valid_indices = scores_valid.index
        self.valid_tickers = self.tickerlist[self.valid_indices]
        return (scores_valid)
    
    def set_valid_df(self):
        self.score_df = self.keep_valid()
    
    def standardise_df(self):
        score_df = self.get_score_df()
        score_df = (score_df - score_df.mean()) / score_df.std()
        return(score_df)
    
    def min_max_df(self):
        score_df = self.get_score_df()
        score_df = (score_df - score_df.min()) / (score_df.max()-score_df.min())
        return(score_df)
    
    def get_rank_df(self, method = 'min'):
        scores_valid = self.keep_valid()
        scores_ranks = scores_valid.copy()
        scores_ranks = scores_ranks.rank(method = method)
        return(scores_ranks)
    
    def ticker_sector(self, NCAIS = True):
        NCAIS_map = {
        '11':	'AGRI Agriculture, Forestry, Fishing and Hunting',
        '21':	'MINI Mining, Quarrying, and Oil and Gas Extraction',
        '22':	'UTIL Utilities',
        '23':	'CONS Construction',
        '31-33':	'MANU Manufacturing',
        '41/42':	'WHOL Wholesale Trade',
        '44-45':	'RETA Retail Trade',
        '48-49':	'TRAN Transportation and Warehousing',
        '51':	'INFO Information',
        '52':	'FINA Finance and Insurance',
        '53':	'REAL Real Estate and Rental and Leasing',
        '54':	'PROF Professional, Scientific, and Technical Services',
        '55':	'MANA Management of Companies and Enterprises',
        '56':	'ADMI Admini. & Support & Waste Management & Remediation Services',
        '61':	'EDUC Educational Services',
        '62':	'HEAL Health Care and Social Assistance',
        '71':	'ARTS Arts, Entertainment, and Recreation',
        '72':	'ACCO Accommodation and Food Services',
        '81':	'OTHE Other Services (except Public Administration)',
        '91/92':	'PUBL Public Administration'
        }
        if NCAIS:
            RE_sector = self.RE.copy()
            RE_sector['Sector'] = RE_sector['NCAIS'].map(NCAIS_map)
            RE_sector = RE_sector.rename(columns = {'Constituent RIC':'Tag'})  
        else:
            RE_sector = self.RE.copy().rename(columns = {'Constituent RIC':'Tag','Industry':'Sector'})   
        self.sector_df = RE_sector[['Tag', 'Sector']]
        
    def valid_ticker_sector(self):
        self.ticker_sector()
        self.valid_sector_df = self.sector_df.copy().loc[self.valid_indices]
        
    def get_valid_sector_df(self):
        return self.valid_sector_df
    
    def worst_score(self, scores_ranks, n_classes):
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
        reverse_rank = lambda x: 334-x
        
        for agency in ranks.columns:
            ranks[agency] = ranks[agency].map(reverse_rank)
            ranks[agency] = ranks[agency].map(mapping)
        
        # Worst score across columns
        min_scores = ranks.min(axis = 1)
        return(pd.DataFrame({'Tag':self.valid_tickers,'Score':min_scores}))
    
    def plot_distributions(self, df, dist_type, binwidth = 0.1):
        cmap = 'GnBu_d'
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(4, 1, figsize=(10, 16), constrained_layout=True)

        for (i, agency) in enumerate(df.columns):
            sns.histplot(data=df, x=agency, hue=agency, palette=cmap,
                         ax=ax[i], multiple='stack', binwidth = binwidth, legend=False)
            ax[i].set_title(str(agency) + ' ' + dist_type + ' scores distribution')

        plt.savefig("Figures/" + dist_type + "_distributions.png")