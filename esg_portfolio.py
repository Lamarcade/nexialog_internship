# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:42:47 2024

@author: LoïcMARCADET
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from Portfolio import Portfolio
import pandas as pd
from collections import defaultdict
from matplotlib.ticker import FuncFormatter

class ESG_Portfolio(Portfolio):   
    def __init__(self, mu,sigma, rf, ESGs, short_sales = True, sectors = None, tickers = None, rf_params = False):
        """
        Initializes an ESG_Portfolio instance.

        Parameters:
            mu (array-like): Expected returns of the assets.
            sigma (array-like): Covariance matrix of the asset returns.
            rf (float): Risk-free rate.
            ESGs (array-like): ESG scores for the assets.
            short_sales (bool, optional): If True, allows short selling. Defaults to True.
            sectors (DataFrame, optional): DataFrame containing sector information. Defaults to None.
            tickers (list, optional): List of asset tickers. Defaults to None.
            rf_params (bool, optional): If True, includes a risk-free asset. Defaults to False.
        """        
        super().__init__(mu,sigma, rf, short_sales, sectors, tickers, rf_params)
        self.ESGs = ESGs
        
    def get_ESG(self, weights):
        """
        Compute the ESG score of a portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            float: Weighted ESG score of the portfolio.
        """
        if self.rf_params:
            # The risk-free asset does not have an ESG score
            if weights[0] < 0.99:
                return weights[1:].dot(self.ESGs) / sum(weights[1:])
            else:
                return 0
        else:
            return weights.dot(self.ESGs) / sum(weights)

    def set_ESGs(self, ESGs):
        """
        Set the ESG scores for the assets.

        Parameters:
            ESGs (array-like): ESG scores.
        """        
        self.ESGs = ESGs
        
    def set_ESG_pref(self, ESG_pref):
        """
        Set the ESG preference function (see Pedersen 2022).

        Parameters:
            ESG_pref (function): ESG preference function.
        """        
        self.ESG_pref = ESG_pref
    
    def mvESG(self, weights, risk_aversion):
        """
        Calculate the mean-variance ESG utility of a portfolio.

        Parameters:
            weights (array-like): Asset weights in the portfolio.
            risk_aversion (float): Risk aversion coefficient.

        Returns:
            float: Mean-variance ESG utility value.
        """
        return risk_aversion/2 * self.get_variance(weights) - self.get_return(weights) - self.ESG_pref(self.get_ESG(weights))
    
    def ESG_constraint(self, min_ESG):
        """
        Create an ESG constraint for the optimization.

        Parameters:
            min_ESG (float): Minimum ESG score required.

        Returns:
            dict: Constraint dictionary for optimization.
        """        
        return({'type': 'ineq', 'fun': lambda w: self.get_ESG(w) - min_ESG})
    
    def asset_constraint(self, bound, is_min = True):
        """
        Create an asset weight constraint for the optimization.

        Parameters:
            bound (float): Constraint bound.
            is_min (bool, optional): If False, applies a maximum weight constraint. Defaults to True.

        Returns:
            dict: Constraint dictionary for optimization.
        """
        if is_min:
            return({'type': 'ineq', 'fun': lambda w: w - bound})
        else:
            return({'type': 'ineq', 'fun': lambda w: bound - w})
        
    def correspondance(self, weights):
        """
        Calculate the sector weights given asset weights.

        Parameters:
            weights (array-like): Asset weights in the portfolio.

        Returns:
            DataFrame: Sector weight correspondance.
        """
        association = pd.DataFrame({'Ticker': self.sectors['Tag'], 'Sector': self.sectors['Sector'], 'Weight': weights})
        corres = association.groupby('Sector')['Weight'].sum()
        return(corres)
        
    def sector_constraint(self, bound, is_min = True):
        """
        Create a sector constraint for the optimization.

        Parameters:
            bound (float): Constraint bound.
            is_min (bool, optional): If False, applies a maximum weight constraint. Defaults to True.

        Returns:
            dict: Constraint dictionary for optimization.
        """
        if is_min:          
            return({'type': 'ineq', 'fun': lambda w: self.correspondance(w) - bound})
        else:
            return({'type': 'ineq', 'fun': lambda w: bound - self.correspondance(w)})
    
    def optimal_portfolio_ESG(self, min_ESG, input_bounds = None):
        """
        Calculate the optimal portfolio (Sharpe maximizer) for the given ESG constraints.

        Parameters:
            min_ESG (float): Minimum portfolio ESG score required.
            input_bounds (list, optional): Bounds for the weights. Defaults to None.

        Returns:
            array: Optimal weights.
        """        
        initial_weights = self.init_weights()
        boundaries = self.bounds(input_bounds)
        constraints = [self.weight_constraint(), self.ESG_constraint(min_ESG)]
        
        result = minimize(self.neg_sharpe, x0 = initial_weights, method='SLSQP', 
                          constraints=constraints, 
                          bounds=boundaries)
        return result.x
    
    def efficient_frontier_ESG(self, low_ESG, up_ESG, interval = 1):
        """
        Calculate the efficient frontier for the given ESG constraints.

        Parameters:
            low_ESG (float): Lower bound for ESG scores.
            up_ESG (float): Upper bound for ESG scores.
            interval (int, optional): Step size for ESG scores. Defaults to 1.

        Returns:
            tuple: Sharpe ratios and ESG scores.
        """        
        sharpes, ESG_list = [], []
        for min_ESG in np.arange(low_ESG, up_ESG, interval):
            weights = self.optimal_portfolio_ESG(min_ESG)
            sharpes.append(self.get_sharpe(weights))
            ESG_list.append(min_ESG)
        return sharpes, ESG_list
    
    def diversification_ESG(self, low_ESG, up_ESG, interval = 1):
        """
        Calculate the diversification ratio for given ESG constraints.

        Parameters:
            low_ESG (float): Lower bound for ESG scores.
            up_ESG (float): Upper bound for ESG scores.
            interval (int, optional): Step size for ESG scores. Defaults to 1.

        Returns:
            tuple: Diversification ratios and ESG scores.
        """
        DRs, ESG_list = [], []
        for min_ESG in np.arange(low_ESG, up_ESG, interval):
            weights = self.optimal_portfolio_ESG(min_ESG)
            DRs.append(self.diversification_ratio(weights))
            ESG_list.append(min_ESG)
        return DRs, ESG_list
        
    def find_efficient_assets(self, low_ESG, up_ESG, interval = 1, criterion = 0.001):
        """
        Find the efficient assets for given ESG constraints.

        Parameters:
            low_ESG (float): Lower bound for ESG scores.
            up_ESG (float): Upper bound for ESG scores.
            interval (int, optional): Step size for ESG scores. Defaults to 1.
            criterion (float, optional): Minimum weight criterion. Defaults to 0.001.

        Returns:
            tuple: Indices of efficient assets, ESG scores, and Sharpe ratios.
        """
        indices, ESG_list = [], []
        sharpes = []
        for min_ESG in np.arange(low_ESG, up_ESG, interval):
            weights = self.optimal_portfolio_ESG(min_ESG)
            sharpes.append(self.get_sharpe(weights))
            new_indices = [i for i in range(len(weights)) if weights[i] > criterion]
            ESG_list.append(min_ESG)
            indices.extend(x for x in new_indices if x not in indices and x!= 0)
        return np.sort(indices), ESG_list, sharpes        
    
    def plot_ESG_frontier(self,sharpes, ESG_list, savefig = True, score_source = None, new_fig = True, eng = True):
        """
        Plot the Sharpe-ESG frontier.

        Parameters:
            sharpes (array-like): Sharpe ratios.
            ESG_list (array-like): ESG scores.
            savefig (bool, optional): If True, saves the figure. Defaults to True.
            score_source (str, optional): Source of the ESG scores. Defaults to None.
            new_fig (bool, optional): If True, creates a new figure. Defaults to True.
            eng (bool, optional): If True, uses English labels. Defaults to True.
        """
        if new_fig:
            self.new_figure()
        if eng:
            self.ax.plot(ESG_list, sharpes, label ='Maximum Sharpe ratio', marker='o', linestyle='-')
        else:
            self.ax.plot(ESG_list, sharpes, label ='Ratio de Sharpe optimal', marker='o', linestyle='-')
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
        if eng:
            title = 'ESG constraints impact on Sharpe ratio with ' + str(n_risky) + ' risky assets'
            if not(self.short_sales):
                title += ", no short"
            if score_source is not None:
                title += ', ' + score_source + ' scores'
            self.ax.set_title(title)
            self.ax.set_xlabel('ESG minimal score')
            self.ax.set_ylabel('Sharpe ratio')
        else:
            title = 'Impact des contraintes ESG sur le ratio de Sharpe, ' + str(n_risky) + ' actifs risqués'
            if not(self.short_sales):
                title += ", pas de short"
            if score_source is not None:
                title += ', scores ' + score_source 
            self.ax.set_title(title)
            self.ax.set_xlabel('Score ESG minimum')
            self.ax.set_ylabel('Ratio de Sharpe')
        self.ax.grid(True)
        self.ax.legend()
        
        # Not the same figure anymore
        self.existing_plot = False
        
        self.make_title_save("_ESG_Sharpe_", n_risky, savefig, score_source) 


    def plot_constrained_frontier(self, risks, returns, marker_size = 1, ESG_min_level = 0, sector_min = 0, sector_max = 0, title = '_frontiers_', savefig = False, score_source = None, eng = True):
        """
        Plot the efficient frontier for given constraints.

        Parameters:
            risks (array-like): Array of portfolio risks.
            returns (array-like): Array of portfolio returns.
            marker_size (int, optional): Size of the marker for the plot. Defaults to 1.
            ESG_min_level (float, optional): Minimum ESG level constraint. Defaults to 0.
            sector_min (float, optional): Minimum sector weight constraint. Defaults to 0.
            sector_max (float, optional): Maximum sector weight constraint. Defaults to 0.
            title (str, optional): Title for saving the figure. Defaults to '_frontiers_'.
            savefig (bool, optional): If True, saves the figure. Defaults to False.
            score_source (str, optional): Source of the ESG scores. Defaults to None.
            eng (bool, optional): If True, uses English labels. Defaults to True.
        """
        if eng:
            lbl = 'EF'
            if not(self.short_sales): 
                lbl += ' no short'
            if ESG_min_level:
                lbl += ', min ESG of ' + str(ESG_min_level)
            if sector_min:
                lbl += ', min weight per sector ' + str(sector_min)
            if sector_max:
                lbl += ', max weight per sector ' + str(sector_max)
        else:
            lbl = 'FE'
            if ESG_min_level:
                lbl += ', rang ESG min de ' + str(ESG_min_level)
            if sector_min:
                lbl += ', poids secteur mini ' + str(sector_min)
            if sector_max:
                lbl += ', poids secteur maxi ' + str(sector_max)
                
        self.ax.plot(risks, returns, linestyle = '--', label = lbl)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
        
        if not(self.existing_plot):
            if eng:
                ax_title = 'ESG Constraints impact on EF with ' + str(n_risky) + ' risky assets'
                if score_source is not None:
                    ax_title += ', ' + score_source + ' scores'
                self.ax.set_title(ax_title)
                self.ax.set_xlabel('Risk')
                self.ax.set_ylabel('Return')
            else:
                ax_title = 'Impact des contraintes ESG sur la FE avec ' + str(n_risky) + ' actifs risqués'
                if score_source is not None:
                    ax_title += ', scores ' + score_source
                self.ax.set_title(ax_title)
                self.ax.set_xlabel('Risque')
                self.ax.set_ylabel('Rendement')
            self.ax.grid(True)
            self.existing_plot = True
            
        self.ax.legend()

        self.make_title_save(title, n_risky, savefig, score_source)
        
    def plot_general_frontier(self, risks, returns, fig_label, fig_title, xlabel, ylabel, marker_size = 1, new_fig = True, save = True):
        """
        Plot a general frontier.

        Parameters:
            risks (array-like): Array of portfolio risks (or other statistics).
            returns (array-like): Array of portfolio returns (or other statistics).
            fig_label (str): Label for the plot.
            fig_title (str): Title of the figure.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            marker_size (int, optional): Size of the marker for the plot. Defaults to 1.
            new_fig (bool, optional): If True, creates a new figure. Defaults to True.
            save (bool, optional): If True, saves the figure. Defaults to True.
        """
        if new_fig:
            self.new_figure()
        self.ax.plot(risks, returns, linestyle = '--', label = fig_label)
        
        if not(self.existing_plot):
            self.ax.set_title(fig_title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.grid(True)
            self.existing_plot = True
            
        self.ax.legend()
        
        plt.savefig('Figures/' + fig_title + '.png')
        
    def plot_sectors_composition(self, min_ESG, save = False, source = None, min_visible = 0.01):
        """
        Plot the sector composition of the tangent portfolio.

        Parameters:
            min_ESG (float): Minimum ESG score.
            save (bool, optional): If True, saves the figure. Defaults to False.
            source (str, optional): Source of the ESG scores. Defaults to None.
            min_visible (float, optional): Minimum weight to be visible in the plot. Defaults to 0.01.
        """
        self.new_figure(fig_size = (12,6))
        ordered_composition = self.sectors_composition.copy().sort_values(by = "Weight")
        
        # Only keep visible weights for the graph
        valid_composition = ordered_composition[abs(ordered_composition['Weight']) >= min_visible]

        sns.barplot(data = valid_composition, x = "Acronym", y = "Weight", hue = "Acronym", palette = "viridis", orient = 'v')
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            self.ax.set_title('Tangent portfolio sector composition, {n_risky} assets & ESG min of {min_ESG}'.format(n_risky = n_risky, min_ESG = min_ESG))
            self.ax.set_xlabel('Sector')
            self.ax.set_ylabel('Weight')
            self.ax.grid(True)
            self.existing_plot = True
        
        self.make_title_save("_CompositionESG{mini}_".format(mini = min_ESG), n_risky, save, source)
           
        
    def plot_composition_change(self, low_constraint, up_constraint, save = False, source = None):
        """
        Plot the change in sector composition between two ESG constraints.

        Parameters:
            low_constraint (float): Lower ESG constraint.
            up_constraint (float): Upper ESG constraint.
            save (bool, optional): If True, saves the figure. Defaults to False.
            source (str, optional): Source of the ESG scores. Defaults to None.
        """
        self.new_figure(fig_size = (12,6))
        tangent_weights = self.optimal_portfolio_ESG(low_constraint)
        self.set_sectors_composition(tangent_weights)
        valid_composition = self.sectors_composition.copy()
        #valid_composition = valid_composition[valid_composition['Weight'] != 0]
        
        tangent_weights = self.optimal_portfolio_ESG(up_constraint)
        self.set_sectors_composition(tangent_weights)
        valid_composition['New_Weight'] = self.sectors_composition['Weight']
        valid_composition['Evolution'] = valid_composition['New_Weight'] - valid_composition['Weight']
        
        valid_composition = valid_composition.sort_values(by = "Evolution")
        self.evolution = valid_composition
        
        # Only keep a sector if there is a noticeable change
        valid_composition = valid_composition[abs(valid_composition['Evolution']) >= 0.01]
        
        sns.barplot(data = valid_composition, x = "Acronym", y = "Evolution", hue = "Acronym", palette = 'viridis', orient = 'v')
        self.ax.set_title('Tangent weights evolution between a minimal ESG of {mini} and {maxi}'.format(mini = low_constraint, maxi = up_constraint))
        self.ax.set_xlabel('Sector')
        self.ax.set_ylabel('Weight')
        self.ax.grid(True)
        self.existing_plot = True
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1

        self.make_title_save("_EndEvolution_{mini}to{maxi}_".format(mini = low_constraint, maxi = up_constraint), n_risky, save, source)
     
    def get_sectors_evolution(self, ESG_range, save = False, source = None):
        """
        Get the evolution of sector weights over a range of ESG constraints.

        Parameters:
            ESG_range (array-like): Array of ESG constraint values.
            save (bool, optional): If True, saves the figure. Defaults to False.
            source (str, optional): Source of the ESG scores. Defaults to None.

        Returns:
            dict: Dictionary of sector weights over the ESG range.
        """
        count, num_iters = 1, len(ESG_range)
        sectors_weights = {}
        
        for min_ESG in ESG_range:
            if not(count % 2):
                print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = num_iters))
            tangent_weights = self.optimal_portfolio_ESG(min_ESG)
            self.set_sectors_composition(tangent_weights)
            
            valid_composition = self.sectors_composition.copy()
            for weight, acronym in zip(valid_composition['Weight'],valid_composition['Acronym']):
                if acronym not in sectors_weights:
                    sectors_weights[acronym] = []
                sectors_weights[acronym].append(weight)
            count += 1
        return(sectors_weights)
    
    def sectors_evolution_from_tickers(self, tickers_weights, sectors_df):
        """
        Calculate the sector weights from tickers' weights.

        Parameters:
            tickers_weights (dict): Dictionary of tickers' weights.
            sectors_df (DataFrame): DataFrame containing tickers and their respective sectors.

        Returns:
            dict: Dictionary of sector weights.
        """       
        def add_lists(list1, list2):
            # Extend the shorter list with zeros
            if len(list1) > len(list2):
                list2.extend([0] * (len(list1) - len(list2)))
            elif len(list2) > len(list1):
                list1.extend([0] * (len(list2) - len(list1)))
            # Return the element-wise sum of the two lists
            return [x + y for x, y in zip(list1, list2)]
        #sectors_weights = {}
        matching_sectors = sectors_df[sectors_df['Tag'].isin(tickers_weights.keys())]
        matching_sectors['Acronym'] = matching_sectors['Sector'].str[:4]
        
        ticker_acronym = dict(zip(matching_sectors['Tag'], matching_sectors['Acronym']))
        
        sectors_weights = defaultdict(list)

        for ticker, weights in tickers_weights.items():
            sector = ticker_acronym.get(ticker)
            
            
            sectors_weights[sector] = add_lists(sectors_weights[sector], weights)

        return(sectors_weights)
    
    
    def get_evolution(self, ESG_range):
        """
        Compute the evolution of asset weights over a range of ESG constraints.

        Parameters:
            ESG_range (list of float): A list of ESG constraint values to iterate over.

        Returns:
            dict: A dictionary where keys are asset tickers and values are lists of weights 
                corresponding to each ESG constraint in ESG_range.
        """  
        count, num_iters = 1, len(ESG_range)
        full_weights = {}
        
        for min_ESG in ESG_range:
            if not(count % 2):
                print('Iteration number {count} out of {num_iters}'.format(count = count, num_iters = num_iters))
            tangent_weights = self.optimal_portfolio_ESG(min_ESG)
            
            for weight, ticker in zip(tangent_weights,self.tickers):
                if ticker not in full_weights:
                    full_weights[ticker] = []
                full_weights[ticker].append(weight)
            count += 1
        return(full_weights)
    
    def complete_weights_lists(self, weights_dict):
        """
        Ensure all weight lists in the weights dictionary have the same length by appending zeros 
        where necessary. Also calculates and appends the risk-free asset values.

        Parameters:
            weights_dict (dict): A dictionary where keys are asset tickers and values are lists of weights.

        Returns:
            dict: The input dictionary with all lists extended to the same length and a new entry 
                for risk-free asset values.
        """
        max_length = max(len(values) for values in weights_dict.values())
        for key, values in weights_dict.items():
            while len(values) < max_length:
                values.append(0)
                
        rifa_values = []
        for i in range(max_length):
            sum_values = sum(weights_dict[key][i] for key in weights_dict)
            rifa_values.append(1 - sum_values)

        weights_dict['RIFA'] = rifa_values
        return weights_dict
    
    def plot_asset_evolution(self, ESG_range, sectors_df, save = False, source = None, min_weight = 0.001, assets_weights = None, xlabel = "ESG constraint", eng = True):
        """
        Plot the evolution of asset weights over a range of ESG constraints.

        Parameters:
            ESG_range (list of float): A list of ESG constraint values.
            sectors_df (DataFrame): DataFrame containing sector information for assets.
            save (bool, optional): Whether to save the plot. Default is False.
            source (str, optional): ESG provider for the plot title. Default is None.
            min_weight (float, optional): Minimum weight threshold for assets to be included in the plot. Default is 0.001.
            assets_weights (dict, optional): Pre-computed asset weights. If None, weights will be computed. Default is None.
            xlabel (str, optional): Label for the x-axis. Default is "ESG constraint".
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
        """
        self.new_figure()
        if assets_weights is None:
            assets_weights = self.get_evolution(ESG_range, save, source)

        matching_sectors = sectors_df[sectors_df['Tag'].isin(assets_weights.keys())]
        matching_sectors['Acronym'] = matching_sectors['Sector'].str[:4]
        
        ticker_acronym = dict(zip(matching_sectors['Tag'], matching_sectors['Acronym']))
        
        lss2 = ['solid', 'dotted', 'dashed', 'dashdot']
        i = 0
        for ticker, weights in assets_weights.items():
            acronym = ticker_acronym.get(ticker)
            if max(weights) >= min_weight:
                lines = self.ax.plot(ESG_range[:len(weights)], weights, label=ticker + ' (' + acronym + ')')
                lines[0].set_linestyle(lss2[i//10])
                i+=1
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
        
        figtitle = 'Asset weights (>= ' + str(min_weight) + ') depending on the ESG constraint'
        if not(eng):
            figtitle = "Poids d'actifs (>= " + str(min_weight) + ') selon la contrainte ESG'
        if source is not None:
            figtitle += ', ' + source
        self.ax.set_title(figtitle)
        self.ax.set_xlabel(xlabel)
        if eng:
            self.ax.set_ylabel('Asset weight')
        else:
            self.ax.set_ylabel("Poids d'actifs")
        self.ax.legend(bbox_to_anchor=(1.3, 1))
        self.make_title_save("_AssetEvolution_{mini}to{maxi}_".format(mini = round(min(ESG_range),2), maxi = round(max(ESG_range),2)), n_risky, save, source, bbox_param = 'tight')
    
        
    def plot_sector_evolution(self, ESG_range, save = False, source = None, min_weight = 0.01, sectors_weights = None, xlabel = "ESG constraint", eng = True):
        """
        Plot the evolution of sector weights over a range of ESG constraints.

        Parameters:
            ESG_range (list of float): A list of ESG constraint values.
            save (bool, optional): Whether to save the plot. Default is False.
            source (str, optional): ESG provider for the plot title. Default is None.
            min_weight (float, optional): Minimum weight threshold for sectors to be included in the plot. Default is 0.01.
            sectors_weights (dict, optional): Pre-computed sector weights. If None, weights will be computed. Default is None.
            xlabel (str, optional): Label for the x-axis. Default is "ESG constraint".
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
        """
        self.new_figure()
        if sectors_weights is None:
            sectors_weights = self.get_sectors_evolution(ESG_range, save, source)
        
        for sector, weights in sectors_weights.items():
            if max(weights) >= min_weight:
                self.ax.plot(ESG_range[:len(weights)], weights, label=sector)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
        
        axtitle = 'Sector weights (>= ' + str(min_weight) + ') depending on the ESG constraint'
        if not(eng):
            axtitle = 'Poids sectoriels (>= ' + str(min_weight) + ') selon la contrainte ESG'
        if source is not None:
            axtitle += ', ' + source
        self.ax.set_title(axtitle)
        self.ax.set_xlabel(xlabel)
        if eng:
            self.ax.set_ylabel('Sector weight')
        else:
            self.ax.set_ylabel('Poids sectoriel')
        self.ax.legend()
        self.make_title_save("_Evolution_{mini}to{maxi}_".format(mini = min(ESG_range), maxi = max(ESG_range)), n_risky, save, source)
        
    def plot_sharpe_exclusion(self, sharpes, threshold_list, save, source, eng = True, esg_levels = False, esgs = None):      
        """
        Plot the Sharpe ratio as a function of the number of excluded worst ESG assets.

        Parameters:
            sharpes (list of float): List of Sharpe ratios.
            threshold_list (list of int): List of exclusion thresholds.
            save (bool): Whether to save the plot.
            source (str): ESG provider for the plot title.
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
            esg_levels (bool, optional): If True, adds a secondary x-axis with ESG levels. Default is False.
            esgs (list of float, optional): List of ESG values. Default is None.
        """
        self.ax.plot(threshold_list, sharpes, label = source)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            if eng:
                self.ax.set_title('Sharpe ratio depending on the exclusion of assets')
                self.ax.set_xlabel('Number of worst ESG stocks excluded')
                self.ax.set_ylabel('Sharpe ratio')
            else:
                self.ax.set_title("Ratio de Sharpe du portefeuille selon le nombre d'actifs exclus")
                self.ax.set_xlabel("Nombre d'actifs exclus")
                self.ax.set_ylabel('Ratio de Sharpe')
            self.ax.grid(True)
            self.existing_plot = True
        
        if esg_levels:
            self.ax2 = self.ax.twiny()
            #self.ax2.set_xticks(self.ax.get_xticks())
            self.ax2.set_xbound(self.ax.get_xbound())
            self.ax2.set_xticklabels(esgs)
            self.ax2.set_xlabel("ESG du portefeuille optimal")
        
        self.ax.legend()
        self.make_title_save("_Sharpe_Exclusion_ESG_", n_risky, save) 
    
    def plot_esg_exclusion(self, ESGs, threshold_list, save, source, eng = True):      
        """
        Plot the portfolio ESG score as a function of the number of excluded worst ESG assets.

        Parameters:
            ESGs (list of float): List of ESG scores.
            threshold_list (list of int): List of exclusion thresholds.
            save (bool): Whether to save the plot.
            source (str): ESG provider for the plot title.
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
        """        
        self.ax.plot(threshold_list, ESGs, label = source)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            if eng:
                self.ax.set_title('Portfolio ESG depending on the exclusion of assets')
                self.ax.set_xlabel('Number of worst ESG stocks excluded')
                self.ax.set_ylabel('ESG')
            else:
                self.ax.set_title("ESG du portefeuille selon le nombre d'actifs")
                self.ax.set_xlabel("Nombre d'actifs exclus")
                self.ax.set_ylabel('ESG')
            self.ax.grid(True)
            self.existing_plot = True
        
        self.ax.legend()
        self.make_title_save("_ESG_Exclusion_", n_risky, save) 
        
    def plot_esg_exclusions(self, ESGs_list, threshold_list, save = False, eng = True, gaussian = False):
        """
        Plot the portfolio ESG score as a function of the number of excluded worst ESG assets.

        Parameters:
            ESGs_list (list of list of float): List containing a list of ESG scores correspondin to a confidence interval (if gaussian is False) or the 95% confidence minimum ESG (if gaussian is True).
            threshold_list (list of int): List of exclusion thresholds.
            save (bool, optional): Whether to save the plot.
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
            gaussian (bool, optional): If True, use Gaussian Mixture model (GMM) uncertainty. Default is False.
        """        
        if eng:
            self.ax.plot(threshold_list, ESGs_list[2], label = 'Mean')
            self.ax.fill_between(threshold_list, ESGs_list[0], ESGs_list[1], alpha = .25, color = 'g', label = 'Min-Max ESG')
        else:
            if gaussian:
                self.ax.plot(threshold_list, ESGs_list[0], label = 'ESG du GMM')
                self.ax.fill_between(threshold_list, ESGs_list[1][:len(ESGs_list[0])], ESGs_list[0], alpha = .25, color = 'g', label = 'ESG minimum à 95% de confiance')
            else:
                self.ax.plot(threshold_list, ESGs_list[2], label = 'Moyenne')
                self.ax.fill_between(threshold_list, ESGs_list[0], ESGs_list[1], alpha = .25, color = 'g', label = 'ESG min-max')
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            if eng:
                self.ax.set_title('Portfolio ESG depending on the exclusion of assets')
                self.ax.set_xlabel('Number of worst ESG stocks excluded')
                self.ax.set_ylabel('ESG')
            else:
                self.ax.set_title("ESG du portefeuille et borne inférieure sur l'ESG selon le nombre d'actifs")
                self.ax.set_xlabel("Nombre d'actifs exclus")
                self.ax.set_ylabel('ESG')
            self.ax.grid(True)
            self.existing_plot = True
        
        #self.ax.xaxis.set_major_formatter(PercentFormatter(xmax = 10))
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,y: str(int(x)) + "/10"))
        self.ax.legend(loc = 'upper left')
        self.make_title_save("_ESG_Exclusion_", n_risky, save) 
    
    def plot_sharpe_speed(self, sharpes, ESG_range, save, source, eng = True):      
        """
        Plot the speed of the Sharpe ratio as a function of the ESG constraint.

        Parameters:
            sharpes (list of float): List of Sharpe ratios.
            ESG_range (list of float): List of ESG constraint values.
            save (bool): Whether to save the plot.
            source (str): ESG provider name for the plot title.
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
        """
        speed = np.gradient(sharpes, ESG_range)
        self.ax.plot(ESG_range, speed, label = source)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            if eng:
                self.ax.set_title('Speed of the Sharpe ratio depending on the ESG constraint')
                self.ax.set_xlabel('ESG constraint')
                self.ax.set_ylabel('Speed')
            else:
                self.ax.set_title('Vitesse du ratio de Sharpe selon la contrainte ESG')
                self.ax.set_xlabel('Contrainte ESG')
                self.ax.set_ylabel('Vitesse')
            self.ax.grid(True)
            self.existing_plot = True
        
        self.ax.legend()
        self.make_title_save("_Sharpe_Speed_", n_risky, save)
        
    def plot_sharpe(self, sharpes, ESG_range, save, source, eng = True):     
        """
        Plot the Sharpe ratio as a function of the ESG constraint.

        Parameters:
            sharpes (list of float): List of Sharpe ratios.
            ESG_range (list of float): List of ESG constraint values.
            save (bool): Whether to save the plot.
            source (str): Source information for the plot title.
            eng (bool, optional): If True, use English labels, otherwise use French. Default is True.
        """ 
        self.ax.plot(ESG_range, sharpes, label = source)
        
        n_risky = len(self.mu)
        if self.rf_params:
            n_risky -= 1
            
        if not(self.existing_plot):
            if eng:
                self.ax.set_title('Sharpe ratio depending on the ESG constraint')
                self.ax.set_xlabel('ESG constraint')
                self.ax.set_ylabel('Sharpe')
            else:
                self.ax.set_title('Evolution du ratio de Sharpe selon la contrainte ESG')
                self.ax.set_xlabel('Contrainte ESG')
                self.ax.set_ylabel('Ratio de Sharpe')
            
            self.ax.grid(True)
            self.existing_plot = True
        
        self.ax.legend()
        self.make_title_save("_Sharpe_", n_risky, save)
        
        