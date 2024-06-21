# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:04:09 2024

@author: LoïcMARCADET
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:33:25 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker

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

#%% Create a target variable

load_SM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_SM.load_model('kmeans.pkl')
k_scores = load_SM.get_predictions()
ESGTV3 = load_SM.make_score_2(k_scores, n_classes = 7, gaussian = False)

load_GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_GSM.load_model('gauss.pkl')
full_scores = load_GSM.get_predictions()
ESGTV4 = load_GSM.make_score_2(full_scores, n_classes = 7, gaussian = True)

# =============================================================================
# plt.figure(figsize = (20,6))
# plt.plot(range_number, ESGTV3['Score'], 'bo', label = 'Worst')
# plt.plot(range_number, ESGTV4['Score'], 'go', label = 'Best')
# plt.plot(range_number, ESGTV5['Score'], 'ro', label = 'Average')
# plt.legend()
# plt.show()
# =============================================================================
esg_df = pd.DataFrame({'Tag': valid_tickers, 'KMeans': ESGTV3['Score'], 'GMM': ESGTV4['Score']})
 
#%% Sharpes with exclusion
sharpes_t = [[] for i in range(2)]
ESGs = [[] for i in range(2)]

for i, method in enumerate(esg_df.columns[1:]):
    #%% Get the stock data and keep the companies in common with the target variable
    st = Stocks(path, annual_rf)
    st.process_data()
    st.compute_monthly_returns(drop_index = 60)
    
    # 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
    ESGTV = esg_df[['Tag',method]].rename(columns = {method: 'Score'})
    _ = st.keep_common_tickers(ESGTV, sectors_list)
    #_ = st.keep_common_tickers(ESGTV, sectors_list)
    
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
    
    # =============================================================================
    # #%% New portfolio with only the efficient assets
    # 
    # epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False)
    # #tangent_weights = epf.tangent_portfolio()
    # #tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
    # 
    # epf = epf.risk_free_stats()
    # 
    # #find_sharpes, _ = finder_pf.efficient_frontier_ESG(int(min(stocks_ESG)), int(max(stocks_ESG)), interval = step)
    # sharpes, ESG_list = epf.efficient_frontier_ESG(emin, emax, interval = step)
    # 
    # epf.plot_general_frontier(ESG_list, find_sharpes, fig_label = 'Full assets', fig_title = 'ESG contraints and Sharpe ratio, no short, Su scores', xlabel = None, ylabel = None)
    # epf.plot_general_frontier(ESG_list, sharpes, fig_label = 'Efficient assets', fig_title = 'ESG contraints and Sharpe ratio, no short, Su scores', xlabel = 'ESG constraints', ylabel = 'Sharpe ratio', new_fig = False)
    # 
    # =============================================================================
    #%%
    count_list = range(len(indices))

    for count in count_list:
        est = Stocks(path, annual_rf)
        est.process_data()
        est.compute_monthly_returns(drop_index = 60)
        #ESGTV = esg_df[['Tag',method]].rename({method: 'Score'})
        _ = est.keep_common_tickers(ESGTV, sectors_list)
        
        _, _ = est.select_assets(5)
        stocks_sectors, stocks_ESG = est.keep_assets(indices)
        stocks_sectors, stocks_ESG = est.exclude_assets(count)
        
        est.compute_mean()
        est.compute_covariance()
        mean, _, rf = est.get_mean(), est.get_covariance(), est.get_rf()
        cov = est.covariance_approximation()
    
        xpf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False, sectors = stocks_sectors)
        xpf = xpf.risk_free_stats()
        
        weights_t = xpf.tangent_portfolio()

        sharpes_t[i].append(xpf.get_sharpe(weights_t))
        ESGs[i].append(xpf.get_ESG(weights_t))

#%%        

save = False
xpf.new_figure()
for i, method in enumerate(esg_df.columns[1:]):
    if i == 1:
        save = True
    xpf.plot_sharpe_exclusion(sharpes_t[i], range(len(sharpes_t[i])), save, method, eng = False) 
    
#%%
save = False
xpf.new_figure()
for i, method in enumerate(esg_df.columns[1:]):
    if i == 1:
        save = True
    xpf.plot_esg_exclusion(ESGs[i], range(len(ESGs[i])), save, method, eng = False) 