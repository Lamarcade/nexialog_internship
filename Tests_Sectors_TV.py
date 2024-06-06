# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:23:31 2024

@author: LoïcMARCADET
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:58:41 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker

from scipy.stats import spearmanr

path = "Portefeuille/sp500_stocks_short.csv"
annual_rf = 0.05 # Risk-free rate

#%% Retrieve the scores and compute the ranks 
SG = ScoreGetter('ESG/Scores/')
SG.reduced_df()
scores_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

SG.valid_ticker_sector()

sectors_list = SG.valid_sector_df

# Agencies scores
SG_agencies = ScoreGetter('ESG/Scores/')
SG_agencies.reduced_df()
SG_agencies.set_valid_df()
scores_valid = SG_agencies.get_score_df()
#standard_scores = SG_agencies.standardise_df()
#min_max_scores = SG_agencies.min_max_df()

#SG_agencies.plot_distributions(scores_valid, "")
#SG_agencies.plot_distributions(min_max_scores, "min_max")

agencies_df_list = []
for agency in scores_ranks.columns:
    agencies_df_list.append(pd.DataFrame({'Tag': valid_tickers, 'Score': scores_valid[agency]}))
    
min_max_scores = SG_agencies.min_max_df()
   
#%% Create a target variable

load_SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_SM.load_model('kmeans.pkl')
k_scores = load_SM.get_predictions()
ESGTV3 = load_SM.make_score_2(k_scores, n_classes = 7, gaussian = False)

load_GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_GSM.load_model('gauss.pkl')
full_scores = load_GSM.get_predictions()
ESGTV4 = load_GSM.make_score_2(full_scores, n_classes = 7, gaussian = True)
 
esg_df = pd.DataFrame({'Tag': valid_tickers, 'KMeans': ESGTV3['Score'], 'GMM': ESGTV4['Score']})

#%% Sectors with exclusion
weights_agencies = [{} for i in range(4)]
assets_weights_agencies = [{} for i in range(4)]

for i, method in enumerate(esg_df.columns[1:]):
    #%% Get the stock data and keep the companies in common with the target variable
    st = Stocks(path, annual_rf)
    st.process_data()
    st.compute_monthly_returns(drop_index = 60)
    
    ESGTV = esg_df[['Tag',method]].rename(columns = {method: 'Score'})
    #_ = st.keep_common_tickers(agencies_df_list[i], sectors_list)
    _ = st.keep_common_tickers(ESGTV, sectors_list)
    
    stocks_sectors, stocks_ESG = st.select_assets(5)
    #stocks_ESG = st.restrict_assets(50)
    st.compute_mean()
    st.compute_covariance()
    mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
    cov = st.covariance_approximation()
    
    #%% Build a portfolio with restrictions on the minimal ESG score
    
    finder_pf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False)
    #tangent_weights = epf.tangent_portfolio()
    #tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
    
    finder_pf = finder_pf.risk_free_stats()
    
    step = 1
    count, num_iters = 1, 1 + ((int(max(stocks_ESG))) - int(min(stocks_ESG))) // step
    emin, emax = int(min(stocks_ESG)), int(max(stocks_ESG))
    
    indices, ESG_range, find_sharpes = finder_pf.find_efficient_assets(emin, emax, step, criterion = 10**(-3))
    
    stocks_sectors, stocks_ESG = st.keep_assets(indices)
    st.compute_mean()
    st.compute_covariance()
    mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
    cov = st.covariance_approximation()
    
    count_list = range(len(indices))
    assets_weights = {}
    #sectors_weights = {}
    print('Analysis for {agency}'.format(agency = agency))
    for count in count_list:
        if not(count%2):
            print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = len(indices)))
        est = Stocks(path, annual_rf)
        est.process_data()
        est.compute_monthly_returns(drop_index = 60)
    
        _ = est.keep_common_tickers(ESGTV, sectors_list)
        
        _, _ = est.select_assets(5)
        stocks_sectors, stocks_ESG = est.keep_assets(indices)
        stocks_sectors, stocks_ESG = est.exclude_assets(count)
        
        
        est.compute_mean()
        est.compute_covariance()
        mean, _, rf = est.get_mean(), est.get_covariance(), est.get_rf()
        cov = est.covariance_approximation()
    
        xpf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = stocks_sectors, tickers = est.tickers)
        xpf = xpf.risk_free_stats()
        
        weights_t = xpf.tangent_portfolio()

        #xpf.set_sectors_composition(weights_t)
        
        #valid_composition = xpf.sectors_composition.copy()
        #for weight, acronym in zip(valid_composition['Weight'],valid_composition['Acronym']):
           # if acronym not in sectors_weights:
                #sectors_weights[acronym] = []
            #sectors_weights[acronym].append(weight)
            
        for weight, ticker in zip(weights_t,xpf.tickers):
            if ticker not in assets_weights:
                assets_weights[ticker] = []
            assets_weights[ticker].append(weight)
    #weights_agencies[i] = sectors_weights
    assets_weights_agencies[i] = assets_weights

#%%
complete_sectors = sectors_list.copy()
complete_sectors.loc[-1] = ['RIFA', 'RISK Risk-Free Asset']
complete_sectors.sort_index(inplace = True)
 
#%%
i = 0   
for assets_weights, method in zip(assets_weights_agencies, esg_df.columns[1:]):
    max_length = max([len(assets_weights[ticker]) for ticker in assets_weights])
    
    complete_weights = xpf.complete_weights_lists(assets_weights)

    
    xpf.plot_asset_evolution(range(max_length), complete_sectors, save = True, source = method, min_weight = 0.0001, assets_weights = complete_weights, xlabel = "Number of worst ESG stocks excluded")
    #xpf.plot_asset_evolution(range(max_length), complete_sectors, save = True, source = agency, min_weight = 0.0001, assets_weights = assets_weights, xlabel = "Number of worst ESG stocks excluded")

    sectors_weights = xpf.sectors_evolution_from_tickers(assets_weights, complete_sectors)
    weights_agencies[i] = sectors_weights
    i += 1

#%% 
for i, method in enumerate(esg_df.columns[1:]):
    weights = weights_agencies[i]
    max_length = max([len(weights[acronym]) for acronym in weights])
    xpf.plot_sector_evolution(range(max_length), save = True, source = method, min_weight = 0.001, sectors_weights = weights, xlabel = "Number of worst ESG stocks excluded")

#