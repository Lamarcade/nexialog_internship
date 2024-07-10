# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:58:23 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks

from ScoreGetter import ScoreGetter
from ScoreMaker import ScoreMaker

from scipy.stats import spearmanr

class Tester:
    def __init__(self, path =  "Portefeuille/sp500_stocks_short.csv", annual_rf = 0.05, portfolio = None):
        #%% Retrieve the scores and compute the ranks 
        self.SG = ScoreGetter('ESG/Scores/')
        self.SG.reduced_df()
        self.scores_ranks = self.SG.get_rank_df()
        self.dict_agencies = self.SG.get_dict()
        self.valid_tickers, self.valid_indices = self.SG.get_valid_tickers(), self.SG.get_valid_indices()

        self.SG.valid_ticker_sector()
        self.sectors_list = self.SG.valid_sector_df
        
        self.set_agencies_df_list()
        self.pf = portfolio
        
        self.path = path
        self.annual_rf = annual_rf # Risk-free rate
        
        self.agencies_indices = {0:"MS", 1:"SU", 2:"SP", 3:"RE"}
    
    def set_scores(self):
        SG_agencies = ScoreGetter('ESG/Scores/')
        SG_agencies.reduced_df()
        SG_agencies.set_valid_df()
        self.scores_valid = SG_agencies.get_score_df()
        self.standard_scores = SG_agencies.standardise_df()   
        
    def set_agencies_df_list(self, ranks = True, scores = False, standard = False):
        self.agencies_df_list = []
        for agency in self.scores_ranks.columns:
            if ranks:
                self.agencies_df_list.append(pd.DataFrame({'Tag': self.valid_tickers, 'Score': self.scores_ranks[agency]}))
            elif scores:
                self.agencies_df_list.append(pd.DataFrame({'Tag': self.valid_tickers, 'Score': self.scores_valid[agency]}))
            else:
                self.agencies_df_list.append(pd.DataFrame({'Tag': self.valid_tickers, 'Score': self.standard_scores[agency]}))
    
    def clusters_variables(self, standardize = False):
        load_SM = ScoreMaker(self.scores_ranks, self.dict_agencies, self.valid_tickers, self.valid_indices, 7)
        load_SM.load_model('kmeans.pkl')
        k_scores = load_SM.get_predictions()
        ESGTV3 = load_SM.make_score_2(k_scores, n_classes = 7, gaussian = False)
        self.load_SM = load_SM

        load_GSM = ScoreMaker(self.scores_ranks, self.dict_agencies, self.valid_tickers, self.valid_indices, 7)
        load_GSM.load_model('gauss.pkl')
        full_scores = load_GSM.get_predictions()
        ESGTV4 = load_GSM.make_score_2(full_scores, n_classes = 7, gaussian = True)
        
        self.load_GSM = load_GSM
        
        self.clusters_df_list = []
        if standardize:
            std_ESGTV3 = ESGTV3.copy()
            std_ESGTV3['Score'] = (std_ESGTV3['Score'] - std_ESGTV3['Score'].mean()) / std_ESGTV3['Score'].std()
            std_ESGTV4 = ESGTV4.copy()
            std_ESGTV4['Score'] = (std_ESGTV4['Score'] - std_ESGTV4['Score'].mean()) / std_ESGTV4['Score'].std()
            self.clusters_df_list.append(std_ESGTV3)
            self.clusters_df_list.append(std_ESGTV4)
        else:
            self.clusters_df_list.append(pd.DataFrame({'Tag': self.valid_tickers, 'Score': ESGTV3['Score'].rank(method = 'min')}))
            self.clusters_df_list.append(pd.DataFrame({'Tag': self.valid_tickers, 'Score': ESGTV4['Score'].rank(method = 'min')}))
    
    def harmonized_variables(self):
        # Worst score approach
        ESGTV5, all_ranks = self.SG.worst_score(self.scores_ranks, n_classes = 7, get_all = True)
        ESGTV6 = self.SG.worst_score(self.scores_ranks, n_classes = 7, reverse = True)

        ESGTV7 = pd.DataFrame({'Tag': ESGTV5['Tag'], 'Score': round(all_ranks.mean(axis = 1)).astype(int)})

        std_ESGTV5 = ESGTV5.copy()
        std_ESGTV5['Score'] = (std_ESGTV5['Score']- std_ESGTV5['Score'].mean()) / std_ESGTV5['Score'].std()
        std_ESGTV6 = ESGTV6.copy()
        std_ESGTV6['Score'] = (std_ESGTV6['Score']- std_ESGTV6['Score'].mean()) / std_ESGTV6['Score'].std()
        std_ESGTV7 = ESGTV7.copy()
        std_ESGTV7['Score'] = (std_ESGTV7['Score']- std_ESGTV7['Score'].mean()) / std_ESGTV7['Score'].std()
        
        self.harmonized_df_list = [std_ESGTV5, std_ESGTV6, std_ESGTV7]
    
    def combine_TV_lists(self):
        self.agencies_indices = {0:"MS", 1:"SU", 2:"SP", 3:"RE", 4:"KMeans", 5:"GMM", 6:"Pire", 7:"Meilleur", 8:"Moyen"}
        if self.clusters_df_list is not None:
            self.agencies_df_list.extend(self.clusters_df_list)
        
        if self.harmonized_df_list is not None:
            self.agencies_df_list.extend(self.harmonized_df_list)
    
    def set_portfolio(self, portfolio):
        self.pf = portfolio
       
    def stocks_init(self, provider_index = 3, agencies = True, ESGTV = None, set_pf = True):
        
        # Get the stock data and keep the companies in common with the target variable
        st = Stocks(self.path, self.annual_rf)
        st.process_data()
        st.compute_monthly_returns()

        if ESGTV is not None:
            _ = st.keep_common_tickers(ESGTV, self.sectors_list)
            
        elif agencies:
            _ = st.keep_common_tickers(self.agencies_df_list[provider_index], self.sectors_list)
        else:
            _ = st.keep_common_tickers(self.clusters_df_list[provider_index], self.sectors_list)

        self.stocks_sectors, self.stocks_ESG = st.select_assets(5)
        st.compute_mean()
        st.compute_covariance()
        self.mean, _ , self.rf = st.get_mean(), st.get_covariance(), st.get_rf()
        self.cov = st.covariance_approximation()
        
        self.st = st
        
        if set_pf:
            epf = ESG_Portfolio(self.mean,self.cov,self.rf, self.stocks_ESG, short_sales = False, sectors = self.stocks_sectors)
            self.set_portfolio(epf)
            
    def add_risk_free(self):
        self.pf.risk_free_stats()
    
    def esg_frontiers(self, emin, emax, step, provider_index, save_figure = True): 
        risks, returns, sharpes = self.pf.efficient_frontier(max_std = 0.10, method = 2)
        self.pf.new_figure()
        self.pf.plot_constrained_frontier(risks, returns, eng = False)
        save = False
        count, num_iters = 1, 1 + (emax - emin) // step
    
        for min_ESG in range(emin, emax, step):
            if not(count % 2):
                print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = int(num_iters)))
            risks_new, returns_new, sharpes_new = self.pf.efficient_frontier(max_std = 0.10, method = 2, new_constraints = [self.pf.ESG_constraint(min_ESG)])
            if min_ESG >= (emax-1-step):
                save = save_figure
            self.pf.plot_constrained_frontier(risks_new, returns_new, ESG_min_level = min_ESG, savefig = save, score_source = self.agencies_indices[provider_index], eng = False)
            count += 1
            
    def sector_esg_frontiers(self, provider_index = 3):
        #% Efficient frontier with ESG and sector constraints

        risks, returns, _ = self.pf.efficient_frontier(max_std = 0.10, method = 2)
        self.pf.new_figure()
        self.pf.plot_constrained_frontier(risks, returns, eng = False)

        threshold = 82
        min_sector = 0.03
        print('Constrained frontier for ESG constraint')
        risks_esg, returns_esg, _ = self.pf.efficient_frontier(max_std = 0.10, method = 2, new_constraints = [self.pf.ESG_constraint(threshold)])

        n_sectors = self.stocks_sectors['Sector'].nunique()
        print('Constrained frontier for sectors constraint')
        risks_sectors, returns_sectors, _ = self.pf.efficient_frontier(max_std = 0.10, method = 2, new_constraints = [self.pf.sector_constraint(min_sector*np.ones(n_sectors))])

        print('Constrained frontier for both constraints')
        risks_all, returns_all, _ = self.pf.efficient_frontier(max_std = 0.10, method = 2, new_constraints = [self.pf.ESG_constraint(threshold), self.pf.sector_constraint(min_sector*np.ones(n_sectors))])

        self.pf.plot_constrained_frontier(risks_esg, returns_esg, ESG_min_level = threshold, eng = False)
        self.pf.plot_constrained_frontier(risks_sectors, returns_sectors, sector_min = min_sector, eng = False)
        self.pf.plot_constrained_frontier(risks_all, returns_all, ESG_min_level = threshold, sector_min = min_sector, savefig = True, title = '_ESGSector82_', score_source = self.agencies_indices[provider_index], eng = False)
            
    def sector_evolution(self, emin, emax, step):
        ESG_range = range(emin, emax, step)
        
        for i,agency in enumerate(self.scores_ranks.columns):
            
            self.stocks_init(i)
     
            assets_weights = self.pf.get_evolution(ESG_range)
            self.pf.plot_asset_evolution(range(emin, emax, step), self.stocks_sectors, save = True, source = agency, min_weight = 0.001, assets_weights = assets_weights, xlabel = "Contrainte de rang ESG", eng = False)
            
            sectors_weights = self.pf.sectors_evolution_from_tickers(assets_weights, self.stocks_sectors)
            
            self.pf.plot_sector_evolution(ESG_range, save = True, source = agency, min_weight = 0.001, sectors_weights = sectors_weights, xlabel = "Contrainte de rang ESG", eng = False)

    def sharpe_analysis(self, low_ESG = 0, up_ESG = 1.05, step = 0.01):
        
        save = False
        spearmans = {}
        reduced_df = pd.DataFrame()
        p = len(self.agencies_df_list)
        for i in range(p):
            print(i)
            agency = self.agencies_indices[i]
            self.stocks_init(i, set_pf = False)
            
            reduced_df[agency] = self.stocks_ESG
            
            # Build a portfolio with restrictions on the minimal ESG score
            if i == 0:
                epf = ESG_Portfolio(self.mean,self.cov,self.rf, self.stocks_ESG, short_sales = False, sectors = self.stocks_sectors)

                epf.new_figure(fig_size = (12,12))
                self.set_portfolio(epf)
            
            self.pf.set_ESGs(self.stocks_ESG)
            
            emin, emax = min(self.stocks_ESG), max(self.stocks_ESG)
            sharpes, ESG_list = self.pf.efficient_frontier_ESG(emin, emax, interval = step)
            spearmans[agency] = spearmanr(sharpes, ESG_list).statistic
            if i == p-1:
                save = True
            self.pf.plot_sharpe(sharpes, ESG_list, save = save, source = agency)

        return(spearmans)
    
    def sharpe_exclusion(self):
        
        complete_sectors = self.sectors_list.copy()
        complete_sectors.loc[-1] = ['RIFA', 'RISK Risk-Free Asset']
        complete_sectors.sort_index(inplace = True)
        self.complete_sectors = complete_sectors
        
        p = len(self.agencies_df_list)

        #% Sharpes with exclusion
        sharpes_t = [[] for i in range(p)]

        for i in range(p):
            print(i)
            agency = self.agencies_indices[i]
            
            self.stocks_init(i)
            
            #% Build a portfolio with restrictions on the minimal ESG score
            
            finder_pf = ESG_Portfolio(self.mean,self.cov,self.rf, self.stocks_ESG, short_sales = False)
            #tangent_weights = epf.tangent_portfolio()
            #tangent_risk, tangent_return = epf.get_risk(tangent_weights), epf.get_return(tangent_weights)
            
            finder_pf = finder_pf.risk_free_stats()
            
            step = 1

            emin, emax = int(min(self.stocks_ESG)), int(max(self.stocks_ESG))
            
            indices, ESG_range, find_sharpes = finder_pf.find_efficient_assets(emin, emax, step, criterion = 10**(-3))
            
            stocks_sectors, stocks_ESG = self.st.keep_assets(indices)
            self.st.compute_mean()
            self.st.compute_covariance()
            self.mean, _ , self.rf = self.st.get_mean(), self.st.get_covariance(), self.st.get_rf()
            self.cov = self.st.covariance_approximation()
            
            #%%
            count_list = range(len(indices))

            for count in count_list:
                est = Stocks(self.path, self.annual_rf)
                est.process_data()
                est.compute_monthly_returns()
            
                _ = est.keep_common_tickers(self.agencies_df_list[i], self.sectors_list)
                
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

        #%%

        save = False
        xpf.new_figure()
        
        for i in range(p):
            agency = self.agencies_indices[i]
            if i == (p-1):
                save = True
            xpf.plot_sharpe_exclusion(sharpes_t[i], range(len(sharpes_t[i])), save, agency + ', ' + str(len(sharpes_t[i])) + ' actifs ESG-efficients', eng = False)   
            

    