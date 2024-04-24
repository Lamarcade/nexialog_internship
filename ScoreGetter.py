# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:54:50 2024

@author: LoïcMARCADET
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        
    def get_score_df(self):
        self.make_dict()
        return(pd.DataFrame(self.dict_agencies))
    
    def keep_valid(self):
        score_df = self.get_score_df()
        scores_valid = score_df[~(score_df.isna().any(axis=1))]
        self.valid_indices = scores_valid.index
        self.valid_tickers = self.tickerlist[self.valid_indices]
        return (scores_valid)
    
    def standardise_df(self):
        score_df = self.get_score_df()
        scaler = StandardScaler()
        return(scaler.fit_transform(score_df))
    
    def get_rank_df(self, method = 'min'):
        scores_valid = self.keep_valid()
        scores_ranks = scores_valid.copy()
        scores_ranks = scores_ranks.rank(method = method)
        return(scores_ranks)