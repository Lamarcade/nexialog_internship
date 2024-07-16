# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:54:50 2024

@author: LoïcMARCADET
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ScoreGetter:
    """
    A class to manage and process ESG scores from various sources.

    Attributes:
    MS (pd.DataFrame): DataFrame containing MSCI scores.
    SU (pd.DataFrame): DataFrame containing Sustainalytics scores.
    SP (pd.DataFrame): DataFrame containing S&P scores.
    RE (pd.DataFrame): DataFrame containing Refinitiv scores.
    tickerlist (pd.Series): List of tickers from MSCI scores.
    MSS (pd.DataFrame): Processed MSCI scores.
    SUS (pd.DataFrame): Processed Sustainalytics scores.
    SPS (pd.DataFrame): Processed S&P scores.
    RES (pd.DataFrame): Processed Refinitiv scores.
    dict_agencies (dict): Dictionary of scores from different agencies.
    valid_indices (pd.Index): Indices of valid scores.
    valid_tickers (pd.Series): Valid tickers after filtering.
    sector_df (pd.DataFrame): DataFrame containing sector information.
    valid_sector_df (pd.DataFrame): DataFrame containing valid sector information.
    score_df (pd.DataFrame): DataFrame containing processed score information.
    """
    def __init__(self, path):
        """
        Initializes the ScoreGetter class.

        Parameters:
        path (str): Path to the directory containing the score files.
        """
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
        """
        Sets the processed scores.

        Parameters:
        MSS (pd.DataFrame): Processed MSCI scores.
        SUS (pd.DataFrame): Processed Sustainalytics scores.
        SPS (pd.DataFrame): Processed S&P scores.
        RES (pd.DataFrame): Processed Refinitiv scores.

        Returns:
        None
        """
        self.MSS = MSS
        self.SUS = SUS
        self.SPS = SPS
        self.RES = RES
        
    def get_processed_scores(self):
        """
        Gets the processed scores.

        Returns:
        tuple: Processed MSCI, Sustainalytics, S&P, and Refinitiv scores.
        """
        return(self.MSS,self.SUS, self.SPS, self.RES)
    
    def name_columns(self):
        """
        Renames columns in the score dataframes to standardize them.

        Returns:
        None
        """
        MSS = self.MS.copy().rename(columns = {'Colonne2':'Tag','Colonne1':'Score'})
        SPS = self.SP.copy()
        SUS = self.SU.copy().rename(columns = {'Symbol':'Tag'})
        RES = self.RE.copy().rename(columns = {'Constituent RIC':'Tag','ESG_Score_01/01/2024':'Score'})
        
        self.set_scores(MSS, SUS, SPS, RES)
        
    def transform_scores(self):
        """
        Transforms the scores to a scale where the higher the score, the better the performance.

        Returns:
        None
        """
        self.name_columns()
        MS_order = {'AAA': 6, 'AA': 5, 'A': 4, 'BBB': 3, 'BB': 2, 'B': 1, 'CCC': 0}
        
        self.MSS['Score'] = self.MSS['Score'].map(MS_order)
        
        #A higher score in Sustainalytics is worse
        reverse = lambda x: 100-x
        self.SUS['Score'] = self.SUS['Score'].map(reverse)
        
    def reduce(self):
        """
        Reduces the score dataframes to include only the 'Tag' and 'Score' columns.

        Returns:
        None
        """
        self.MSS = self.MSS[['Tag', 'Score']]
        self.SPS = self.SPS[['Tag', 'Score']]
        self.SUS = self.SUS[['Tag', 'Score']]
        self.RES = self.RES[['Tag', 'Score']]
    
    def reduced_df(self): 
        """
        Transforms and reduces the score dataframes.

        Returns:
        None
        """
        self.transform_scores()
        self.reduce()
        
    def homogen_df(self): 
        """
        Homogenizes the score dataframes without reducing them.

        Returns:
        None
        """
        self.transform_scores()

    def reduced_mixed_df(self): 
        """
        Reduces the score dataframes without transforming them.

        Returns:
        None
        """
        self.name_columns()
        self.reduce()
        
    def make_dict(self):
        """
        Creates a dictionary of scores from different agencies.

        Returns:
        None
        """
        self.dict_agencies = {'MS':  self.MSS['Score'], 'SU': self.SUS['Score'], 
                         'SP' : self.SPS['Score'], 'RE': self.RES['Score']}
    
    def get_dict(self):
        """
        Gets the dictionary of scores from different agencies.

        Returns:
        dict: Dictionary of scores.
        """
        return(self.dict_agencies)
    
    def get_valid_indices(self):
        """
        Gets the indices of valid scores.

        Returns:
        pd.Index: Indices of valid scores.
        """
        return self.valid_indices
    
    def get_valid_tickers(self):
        """
        Gets the valid tickers after filtering.

        Returns:
        pd.Series: Valid tickers.
        """
        return self.valid_tickers
        
    def get_new_df(self):
        """
        Creates a new DataFrame from the dictionary of scores.

        Returns:
        pd.DataFrame: DataFrame of scores from different agencies.
        """
        self.make_dict()
        return(pd.DataFrame(self.dict_agencies))
    
    def get_score_df(self):
        """
        Gets the DataFrame containing processed score information.

        Returns:
        pd.DataFrame: DataFrame of processed scores.
        """
        return self.score_df
    
    def keep_valid(self):
        """
        Filters the score DataFrame to keep only valid scores.

        Returns:
        pd.DataFrame: DataFrame of valid scores.
        """
        score_df = self.get_new_df()
        scores_valid = score_df[~(score_df.isna().any(axis=1))]
        self.valid_indices = scores_valid.index
        self.valid_tickers = self.tickerlist[self.valid_indices]
        return (scores_valid)
    
    def set_valid_df(self):
        """
        Sets the DataFrame of valid scores.

        Returns:
        None
        """
        self.score_df = self.keep_valid()
    
    def standardise_df(self):
        """
        Standardizes the score DataFrame.

        Returns:
        pd.DataFrame: Standardized score DataFrame.
        """
        score_df = self.get_score_df()
        score_df = (score_df - score_df.mean()) / score_df.std()
        return(score_df)
    
    def min_max_df(self):
        """
        Applies min-max scaling to the score DataFrame.

        Returns:
        pd.DataFrame: Min-max scaled score DataFrame.
        """
        score_df = self.get_score_df()
        score_df = (score_df - score_df.min()) / (score_df.max()-score_df.min())
        return(score_df)
    
    def get_rank_df(self, method = 'min'):
        """
        Ranks the scores using a specified method.

        Parameters:
        method (str): Method to use for ranking.

        Returns:
        pd.DataFrame: DataFrame of ranked scores.
        """
        scores_valid = self.keep_valid()
        scores_ranks = scores_valid.copy()
        scores_ranks = scores_ranks.rank(method = method)
        return(scores_ranks)
    
    def ticker_sector(self, NCAIS = True):
        """
        Maps tickers to sectors based on the NCAIS code.

        Parameters:
        NCAIS (bool): If True, use NCAIS codes for mapping.

        Returns:
        None
        """
        NCAIS_map = {
        '11':	'AGRI Agriculture, Forestry, Fishing and Hunting',
        '21':	'MINI Mining, Quarrying, and Oil and Gas Extraction',
        '22':	'UTIL Utilities',
        '23':	'CONS Construction',
        '31-33':	'MANU Manufacturing',
        '41/42':	'WHOL Wholesale Trade',
        '41':	'WHOL Wholesale Trade',
        '42':	'WHOL Wholesale Trade',
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
        '91/92':	'PUBL Public Administration',
        '91':	'PUBL Public Administration',
        '92':	'PUBL Public Administration',
        }
        if NCAIS:
            RE_sector = self.RE.copy()
            RE_sector['Sector'] = RE_sector['NCAIS'].map(NCAIS_map)
            RE_sector = RE_sector.rename(columns = {'Constituent RIC':'Tag'})  
        else:
            RE_sector = self.RE.copy().rename(columns = {'Constituent RIC':'Tag','Industry':'Sector'})   
        self.sector_df = RE_sector[['Tag', 'Sector']]
        
    def valid_ticker_sector(self):
        """
        Filters the sector DataFrame to keep only valid sectors.

        Returns:
        None
        """
        self.ticker_sector()
        self.valid_sector_df = self.sector_df.copy().loc[self.valid_indices]
        
    def get_valid_sector_df(self):
        """
        Gets the DataFrame containing valid sector information.

        Returns:
        pd.DataFrame: DataFrame of valid sectors.
        """
        return self.valid_sector_df
    
    def worst_score(self, scores_ranks, n_classes = 7, reverse = False, get_all = False):        
        """
        Computes the worst or best score across different agencies.

        Parameters:
        scores_ranks (pd.DataFrame): DataFrame of ranked scores.
        n_classes (int): Number of classes to divide the scores into.
        reverse (bool): If True, take the best score instead.
        get_all (bool): If True, return all intermediate results.

        Returns:
        pd.DataFrame: DataFrame of worst scores.
        """
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
        
        # Worst score across columns
        if reverse:
            scores = ranks.max(axis = 1)
        else:
            scores = ranks.min(axis = 1)
        
        res = pd.DataFrame({'Tag':self.valid_tickers,'Score':scores})
        if get_all:
           return res, ranks 
        return(res)
    
    def worst_std_score(self, std_scores, n_classes = 7, reverse = False, get_all = False):
        """
        Computes the worst or best standardized score across different agencies.

        Parameters:
        std_scores (pd.DataFrame): DataFrame of standardized scores.
        n_classes (int): Number of classes to divide the scores into.
        reverse (bool): If True, take the best score instead.
        get_all (bool): If True, return all intermediate results.

        Returns:
        pd.DataFrame: DataFrame of worst standardized scores.
        """
        std_df = std_scores.copy()
        list_scores = list(range(n_classes))
        
        for agency in std_df.columns:
            def mapping(score):
                thresholds = [min(std_df[agency]) + i* (max(std_df[agency]) - min(std_df[agency]))/n_classes for i in range(n_classes)]
                for i in range(len(thresholds)):
                    if score <= thresholds[i]:
                        return(list_scores[i]-1)
                return n_classes-1
            std_df[agency] = std_df[agency].map(mapping)
        
        # Worst score across columns
        if reverse:
            scores = std_df.max(axis = 1)
        else:
            scores = std_df.min(axis = 1)
        
        res = pd.DataFrame({'Tag':self.valid_tickers,'Score':scores})
        if get_all:
           return res, std_df 
        return(res)
    
    def plot_distributions(self, df, dist_type, shrink = 1, n = 4, eng = True):
        """
        Plot the score distributions in a given DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame of  scores.
        dist_type (str): Description of the score distributions.
        shrink (float): Shrink parameter of the seaborn histogram. Defaults to 1
        n (int): Number of distributions in the DataFrame. Defaults to 4
        eng (bool): If True, plot titles and labels in English; otherwise, in French. Defaults to True

        Returns:
        pd.DataFrame: DataFrame of worst standardized scores.
        """
        cmap = 'GnBu_d'
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(n, 1, figsize=(10, 16), constrained_layout=True)

        for (i, agency) in enumerate(df.columns):
            sns.histplot(data=df, x=agency, hue=agency, palette=cmap,
                         ax=ax[i], discrete = True, shrink = shrink, legend=False)
            if eng:
                ax[i].set_title(str(agency) + ' ' + dist_type + ' scores distribution')
            else:
                ax[i].set_title("Distribution des scores " + dist_type + ', ' + str(agency) + ' score')
                ax[i].set_ylabel('Compte')

        plt.savefig("Figures/" + dist_type + "_distributions.png")