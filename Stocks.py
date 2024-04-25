# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:51:09 2024

@author: LoïcMARCADET
"""
import pandas as pd
import numpy as np


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
     
    def keep_common_tickers(self, target_variable):

        #Keep only the companies for which we have monthly returns
        mr_tickers = self.returns.columns.tolist()
        tv_filtered = target_variable.loc[target_variable['Tag'].isin(mr_tickers)]

        self.targetESG = np.array(tv_filtered['Score'].tolist())
        # Keep only the companies for with we have ESG Scores
        tv_tickers = tv_filtered['Tag'].tolist()
        self.returns = self.returns.loc[:, self.returns.columns.isin(tv_tickers)]
        return(self.targetESG)
        
    def restrict_assets(self, n_assets):
        self.n_assets = n_assets
        self.returns = self.returns.iloc[:, :n_assets]
        
        if (self.targetESG is not None):
            self.targetESG = self.targetESG[:n_assets]
            return(self.targetESG)
        
    def compute_mean(self):
        self.mean = np.array(self.returns.mean(axis = 0))
        
    def compute_covariance(self):
        self.covariance = np.array(self.returns.cov())
        
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
    
    