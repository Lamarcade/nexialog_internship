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
    """
    A class to manage stock data, compute returns, and analyze portfolios.
    
    Attributes:
    stocks (pd.DataFrame): DataFrame containing stock data.
    n_assets (int): Number of unique assets in the dataset.
    annual_rf (float): Annual risk-free rate.
    rf (float): Monthly risk-free rate.
    rf_params (bool): Indicator if risk-free asset is included.
    targetESG (np.ndarray): Target ESG scores for assets.
    returns (pd.DataFrame): Monthly returns of assets.
    sectors (pd.DataFrame): Sectors of the assets.
    tickers (np.ndarray): Ticker symbols of the assets.
    index (pd.Index): Indices of the target ESG scores.
    mean (np.ndarray): Mean returns of the assets.
    covariance (np.ndarray): Covariance matrix of the asset returns.
    scov (np.ndarray): Covariance matrix computed using scikit-learn.
    """
    def __init__(self,path, annual_rf):
        """
        Initializes the Stocks class.

        Parameters:
        path (str): Path to the CSV file containing stock data.
        annual_rf (float): Annual risk-free rate.
        """
        self.stocks = pd.read_csv(path)
        self.n_assets = self.stocks['Symbol'].nunique()
        self.annual_rf = annual_rf
        self.rf = (1+annual_rf)**(1/12)-1
        
        # Is the risk-free asset included
        self.rf_params = False
        
        self.targetESG = None

    def process_data(self):
        """
        Processes the stock data by converting the 'Date' column to datetime and pivoting the DataFrame.

        Returns:
        None
        """    
        # Convert 'Date' column to datetime
        self.stocks['Date'] = pd.to_datetime(self.stocks['Date'])
        # Pivot the DataFrame to have symbols as columns
        self.stocks_pivot = self.stocks.pivot(index='Date', columns='Symbol', values='Adj Close')
        
    def compute_monthly_returns(self, n_valid = 50, drop_index = 60):
        """
        Computes monthly returns for each symbol.

        Parameters:
        n_valid (int): Minimum number of valid values required for each column.
        drop_index (int): Index from which to drop rows for better covariance.

        Returns:
        None
        """
        # Calculate monthly returns for each symbol
        monthly_returns = self.stocks_pivot.resample('ME').ffill().pct_change()

        # Get rid of the first row of NaN
        monthly_returns = monthly_returns.iloc[1:]
        
        # After row 60 there are no NaN, which is better for the covariance
        # 10 years of data are left
        if drop_index is not None:
           monthly_returns = monthly_returns.iloc[drop_index:] 
    
        # Drop columns that have less than n_valid valid values
        self.returns = monthly_returns.dropna(axis=1, thresh=n_valid)
     
    def keep_common_tickers(self, target_variable, sectors = None):
        """
        Keeps common tickers between returns and target variable.

        Parameters:
        target_variable (pd.DataFrame): DataFrame containing target variable.
        sectors (pd.DataFrame, optional): DataFrame containing sector information.

        Returns:
        np.ndarray: Array of target ESG scores.
        """
        #Keep only the companies for which we have monthly returns
        mr_tickers = self.returns.columns.tolist()
        tv_filtered = target_variable.loc[target_variable['Tag'].isin(mr_tickers)]

        self.targetESG = np.array(tv_filtered['Score'].tolist())
        if sectors is not None:
            self.sectors = sectors.loc[sectors['Tag'].isin(mr_tickers)]
        
        # Keep only the companies for with we have ESG Scores
        tv_tickers = tv_filtered['Tag'].tolist()
        self.returns = self.returns.loc[:, self.returns.columns.isin(tv_tickers)]
        self.tickers = np.intersect1d(tv_tickers, mr_tickers)
        self.index = tv_filtered.index
        self.n_assets = len(self.tickers)
        return(self.targetESG)
        
    def restrict_assets(self, n_assets = 50):
        """
        Restricts the number of assets to a specified number.

        Parameters:
        n_assets (int): Number of assets to keep.

        Returns:
        np.ndarray: Array of target ESG scores.
        """
        self.n_assets = n_assets
        self.returns = self.returns.iloc[:, :n_assets]
        

        if self.tickers is not None:
            self.tickers = self.tickers[:n_assets]
        if self.index is not None:
            self.index = self.index[:n_assets]
        if self.sectors is not None:
            self.sectors = self.sectors.iloc[:n_assets]                
        if (self.targetESG is not None):
            self.targetESG = self.targetESG[:n_assets]

            return(self.targetESG)
     
    def select_assets(self, n_multi = 5):
        """
        Selects a diversified set of assets.

        Parameters:
        n_multi (int): Number of assets per sector (if there are enough).

        Returns:
        tuple: DataFrame of selected sectors and array of target ESG scores.
        """
        occurrences = {sector:0 for sector in self.sectors['Sector'].dropna().unique()}
        asset_indices, row = pd.Index([0], dtype = int), 1
        occurrences[self.sectors['Sector'].loc[0]] +=1
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
            
        if self.tickers is not None:
            self.tickers = self.tickers[asset_indices.values]
            
        return(self.sectors, self.targetESG)        
    
    def exclude_assets(self, count = 10, threshold = None, ascending_better = True):
        """
        Excludes assets based on ESG scores.

        Parameters:
        count (int): Number of assets to exclude.
        threshold (float): Threshold to determine number of assets to exclude. Overrides count
        ascending_better (bool): If True, lower scores are better.

        Returns:
        tuple: DataFrame of remaining sectors and array of remaining target ESG scores.
        """
        if threshold is not None:
            worst_count = int(threshold * self.n_assets) + 1
        else:
            worst_count = count
        
        if ascending_better:
            best_indices = sorted(range(self.n_assets), key=lambda x: self.targetESG[x])[worst_count:]
        else:
            best_indices = sorted(range(self.n_assets), key=lambda x: self.targetESG[x])[:worst_count]
        best_indices = np.sort(best_indices)
        if not(self.rf_params):
            best_indices +=1

        sectors, targetESG = self.keep_assets(best_indices)
        return(sectors, targetESG)
    
    def keep_assets(self, int_indices):
        """
        Keeps specified assets.

        Parameters:
        int_indices (list): List of indices to keep.

        Returns:
        tuple: DataFrame of kept sectors and array of kept target ESG scores.
        """
        if not(self.rf_params):
            int_indices = [x - 1 for x in int_indices]
        
        self.targetESG = [self.targetESG[i] for i in range(self.n_assets) if i in int_indices]

        
        if self.sectors is not None:
            self.sectors = self.sectors.iloc[int_indices]
            self.index = self.sectors.index
            
        self.n_assets = len(int_indices)
        
        if self.tickers is not None:
            self.tickers = self.tickers[int_indices]
            
        self.returns = self.returns.iloc[:, int_indices]
        return(self.sectors, self.targetESG)
    
    def compute_mean(self):
        """
        Computes mean returns of the assets.

        Returns:
        None
        """
        self.mean = np.array(self.returns.mean(axis = 0))
        
    def compute_covariance(self, bias = False):
        """
        Computes the covariance matrix of asset returns.

        Parameters:
        bias (bool, deprecated): If True, normalize by N instead of N-1.

        Returns:
        None
        """
        self.covariance = np.array(self.returns.cov())
        #self.covariance = np.cov(self.returns, bias = bias)
        
    def scikit_covariance(self):
        """
        Computes the covariance matrix using scikit-learn's empirical_covariance function.

        Returns:
        None
        """
        self.scov = empirical_covariance(self.returns)
    
    def get_mean(self):
        """
        Returns the mean returns of the assets.

        Returns:
        np.ndarray: Mean returns of the assets.
        """
        return self.mean
    
    def get_covariance(self):
        """
        Returns the covariance matrix of the asset returns.

        Returns:
        np.ndarray: Covariance matrix of the asset returns.
        """
        return self.covariance
    
    def get_rf(self):
        """
        Returns the monthly risk-free rate.

        Returns:
        float: Monthly risk-free rate.
        """
        return self.rf
    
    def risk_free_stats(self):
        """
        Adjusts statistics to include the risk-free asset.

        Returns:
        None
        """
        self.mean = np.insert(self.mean,0,self.rf)
        self.n_assets = self.n_assets+1
        covariance_rf = np.zeros((self.n_assets, self.n_assets))
        covariance_rf[1:,1:] = self.covariance
        self.covariance = covariance_rf
        self.rf_params = True
        
    def covariance_approximation(self):
        """
        Approximates the covariance matrix to ensure it is positive semi-definite.

        Returns:
        np.ndarray: Approximated covariance matrix.
        """
        # Get the SPD covariance that minimizes the distance with the actual covariance
        eigenvals, eigenvecs = np.linalg.eigh(self.covariance)
        positive_eigenvals = np.fmax(eigenvals, np.zeros(len(eigenvals)))
        cov_approx = eigenvecs.dot(np.diag(positive_eigenvals)).dot(eigenvecs.T)
        return(cov_approx)
        
    def sector_analysis(self, make_acronym=False):
        """
        Analyzes the sectors of the assets and optionally creates sector acronyms.

        Parameters:
        make_acronym (bool): If True, create 4-letter acronyms for sectors.

        Returns:
        pd.DataFrame: DataFrame with sector analysis.
        """
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

        # Retrieve 4-letter sector acronym
        if make_acronym:
            sorted_df['Acronym'] = sorted_df['Sector'].str.extract(r'([A-Z]{4})')

        return sorted_df
    
    def plot_sectors(self, eng = True):
        """
        Plots the distribution of sectors.

        Parameters:
        eng (bool): If True, plot titles and labels in English; otherwise, in French.

        Returns:
        None
        """
        plt.figure(figsize = (8,12))
        ordered_sectors = self.sector_analysis(make_acronym = True)
        sns.histplot(data=ordered_sectors, y='Acronym', hue='Acronym', legend=False)
        if eng:
            plt.title('Number of companies in each sector according to Refinitiv,' + str(self.n_assets) + ' assets')
        else:
            plt.title("Nombre d'entreprises dans chaque secteur selon Refinitiv," + str(self.n_assets) + " actifs")
            plt.xlabel('Compte')
            plt.ylabel('Secteur')
        figtitle = 'Figures/Sectors_' + str(self.n_assets) + '.png'

        plt.savefig(figtitle, bbox_inches = 'tight')
