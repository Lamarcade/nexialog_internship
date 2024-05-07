# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:51:09 2024

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import empirical_covariance

class Stocks:
    def __init__(self,path, annual_rf):
        self.stocks = pd.read_csv(path)
        self.n_assets = self.stocks.shape[0]
        self.annual_rf = annual_rf
        self.rf = (1+annual_rf)**(1/12)-1
        
        # Is the risk-free asset included
        self.rf_params = False
        
        self.targetESG = None

    def process_data(self):
        # Convert 'Date' column to datetime
        self.stocks['Date'] = pd.to_datetime(self.stocks['Date'])
        # Pivot the DataFrame to have symbols as columns
        self.stocks_pivot = self.stocks.pivot(index='Date', columns='Symbol', values='Adj Close')
        
    def compute_monthly_returns(self, n_valid = 50):
        # Calculate monthly returns for each symbol
        monthly_returns = self.stocks_pivot.resample('ME').ffill().pct_change()

        # Get rid of the first row of NaN
        monthly_returns = monthly_returns.iloc[1:]
   
        # Drop columns that have less than n_valid valid values
        self.returns = monthly_returns.dropna(axis=1, thresh=n_valid)
     
    def keep_common_tickers(self, target_variable, sectors = None):

        #Keep only the companies for which we have monthly returns
        mr_tickers = self.returns.columns.tolist()
        tv_filtered = target_variable.loc[target_variable['Tag'].isin(mr_tickers)]

        self.targetESG = np.array(tv_filtered['Score'].tolist())
        if sectors is not None:
            self.sectors = sectors.loc[sectors['Tag'].isin(mr_tickers)]
        
        # Keep only the companies for with we have ESG Scores
        tv_tickers = tv_filtered['Tag'].tolist()
        self.returns = self.returns.loc[:, self.returns.columns.isin(tv_tickers)]
        return(self.targetESG)
        
    def restrict_assets(self, n_assets = 50):
        self.n_assets = n_assets
        self.returns = self.returns.iloc[:, :n_assets]
        
        if (self.targetESG is not None):
            self.targetESG = self.targetESG[:n_assets]
            if self.sectors is not None:
                self.sectors = self.sectors.iloc[:n_assets]
            return(self.targetESG)
     
    def select_assets(self, n_multi = 5):
        occurrences = {sector:0 for sector in self.sectors['Sector'].dropna().unique()}
        asset_indices, row = pd.Index([0], dtype = int), 1
        while any(s < n_multi for s in occurrences.values()) and (row < len(self.sectors)):
            try:
                new_sector = self.sectors['Sector'].loc[row]
                if occurrences[new_sector] < n_multi and row in self.sectors.index:
                    asset_indices = asset_indices.append(pd.Index([row]))
                    occurrences[new_sector] +=1
            except:
                pass
            row += 1

        self.sectors = self.sectors.loc[asset_indices.values]
        self.index = asset_indices
        self.n_assets = len(asset_indices)
        tags = self.sectors['Tag'].copy()
        #tags = tags.filter(asset_indices)
        self.returns = self.returns[tags]


        if (self.targetESG is not None):
            self.targetESG = self.targetESG[asset_indices.values]
        return(self.sectors, self.targetESG)        
        
     
    def compute_mean(self):
        self.mean = np.array(self.returns.mean(axis = 0))
        
    def compute_covariance(self):
        self.covariance = np.array(self.returns.cov())
        
    def scikit_covariance(self):
        self.scov = empirical_covariance(self.returns)
    
    def get_mean(self):
        return self.mean
    
    def get_covariance(self):
        return self.covariance
    
    def get_rf(self):
        return self.rf
    
    def risk_free_stats(self):
        self.mean = np.insert(self.mean,0,self.rf)
        self.n_assets = self.n_assets+1
        covariance_rf = np.zeros((self.n_assets, self.n_assets))
        covariance_rf[1:,1:] = self.covariance
        self.covariance = covariance_rf
        self.rf_params = True
        
    def covariance_approximation(self):
        
        # Get the SPD covariance that minimizes the distance with the actual covariance
        eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
        positive_eigenvals = np.fmax(eigenvals, np.zeros(len(eigenvals)))
        #print(positive_eigenvals)
        #print(eigenvecs.shape)
        #print(positive_eigenvals.shape)
        cov_approx = eigenvecs.dot(np.diag(positive_eigenvals)).dot(eigenvecs.T)
        return(cov_approx)
        
    def sector_analysis(self, make_acronym=False):
        sectors_df = self.sectors.copy()
        sectors_count = sectors_df['Sector'].value_counts()
        # Create a CategoricalDtype with the correct order
        sorted_sectors = pd.CategoricalDtype(
            categories=sectors_count.index, ordered=True)

        # Create a DataFrame to keep track of the sector count
        sorted_df = sectors_df.sort_values(by='Sector')
        sorted_df['Sector'] = sorted_df['Sector'].astype(sorted_sectors)

        # Sort the DataFrame based on sector count and score
        sorted_df = sorted_df.sort_values(
            by=['Sector'], ascending=True)

        # Retrieve 3-letter sector acronym
        if make_acronym:
            sorted_df['Acronym'] = sorted_df['Sector'].str.extract(r'([A-Z]{3})')

        return sorted_df
    
    def plot_sectors(self):
        plt.figure(figsize = (8,12))
        ordered_sectors = self.sector_analysis(make_acronym = True)
        sns.histplot(data=ordered_sectors, y='Acronym', hue='Acronym', legend=False)
        plt.title('Number of companies in each sector according to Refinitiv,' + str(self.n_assets) + ' assets')
        figtitle = 'Figures/Sectors_' + str(self.n_assets) + '.png'
        plt.savefig(figtitle, bbox_inches = 'tight')
