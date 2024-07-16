# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:29:16 2024

@author: LoïcMARCADET
"""

#%% Libraries  
import pandas as pd
import numpy as np
from scipy.stats import kendalltau, norm
from scipy.optimize import newton, fsolve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#%%


class ScoreMaker:
    """
    A class to manage and process clustering of ESG scores from various sources.

    Attributes:
    ranks (pd.DataFrame): DataFrame containing ranks of ESG scores.
    dict_agencies (dict): Dictionary containing scores from different agencies.
    valid_tickers (pd.Series): Series of valid tickers.
    valid_indices (pd.Index): Indices of valid scores.
    n_classes (int): Number of clusters to form.
    model: The clustering model used for predictions.
    sorted_labels (list): List of sorted cluster labels.
    full_ranks (pd.DataFrame): DataFrame containing full ranks.
    rank_stats (pd.DataFrame): DataFrame containing rank statistics.
    densities (np.ndarray): Densities of the model's predictions.
    clusters_means (pd.Series): Means of the clusters.
    clusters_stds (pd.Series): Standard deviations of the clusters.
    """
    def __init__(self, ranks, dict_agencies, valid_tickers, valid_indices, n_classes):
        """
        Initializes the ScoreMaker class.

        Parameters:
        ranks (pd.DataFrame): DataFrame containing ranks of ESG scores.
        dict_agencies (dict): Dictionary containing scores from different agencies.
        valid_tickers (pd.Series): Series of valid tickers.
        valid_indices (pd.Index): Indices of valid scores.
        n_classes (int): Number of clusters to form.
        """
        self.ranks = ranks.copy()
        self.dict_agencies = dict_agencies
        self.valid_tickers = valid_tickers
        self.valid_indices = valid_indices
        self.n_classes = n_classes
    
        
    def add_class(self, labels):
        """
        Adds cluster labels to the scores.

        Parameters:
        labels (pd.DataFrame): DataFrame containing cluster labels.

        Returns:
        pd.DataFrame: DataFrame with cluster labels added.
        """    
        full_scores = pd.DataFrame(self.ranks, index = self.valid_indices)
        full_scores['labels'] = labels
        return(full_scores)
    
    def cluster_hierarchical(self):
        """
        Performs hierarchical clustering on the ranks.

        Returns:
        pd.DataFrame: DataFrame with hierarchical clustering labels added.
        """
        Agg = AgglomerativeClustering(n_clusters = self.n_classes)
            
        labels = pd.DataFrame(Agg.fit_predict(self.ranks), index = self.valid_indices)   

        return(self.add_class(labels))
    

    def kmeans(self):
        """
        Performs KMeans clustering on the ranks.

        Returns:
        pd.DataFrame: DataFrame with KMeans clustering labels added.
        """
        km = KMeans(n_clusters = self.n_classes)
            
        labels = pd.DataFrame(km.fit_predict(self.ranks), index = self.valid_indices)  
        
        self.model = km
        return(self.add_class(labels))

    def classify_gaussian_mixture(self, class_proportions = None, n_mix = 50):
        """
        Classifies the ranks using Gaussian Mixture Model.

        Parameters:
        class_proportions (dict, optional): Proportions of each class.
        n_mix (int, optional): Number of mixtures to try.

        Returns:
        tuple: DataFrame with Gaussian Mixture Model labels added and the maximum Kendall tau score.
        """
        if not(class_proportions is None):
            # Calculate weights for each cluster based on class proportions
            weights = np.array(list(class_proportions.values()))
            
            # Ensure weights are normalized according to scikit
            weights[-1] = 1-sum(weights[:-1])
        else:
            weights = None
            
        mixtures = []
        taumax = -1
        best_index = 0
        
        for i in range(n_mix): 
            mixtures.append(GaussianMixture(n_components = self.n_classes, weights_init = weights))
            
            mix_labels = mixtures[i].fit_predict(self.ranks)
        
            tau = 0
            for agency in self.dict_agencies:
                #tau += kendalltau(self.dict_agencies[agency][self.valid_tickers], mix_labels, variant = 'c').statistic / len(dict_agencies)
                tau += kendalltau(self.ranks[agency], mix_labels, variant = 'c').statistic / len(self.dict_agencies)

            if tau >= taumax:
                print(tau)
                taumax = tau
                best_index = i

        mixture_labels = pd.DataFrame(mixtures[best_index].predict(self.ranks), index = self.valid_indices)
        
        self.model = mixtures[best_index]
        
        full_scores = self.add_class(mixture_labels)
        return full_scores, taumax
    
    def gmm_1d(self, class_proportions = None, n_mix = 50):
        """
        Classifies the ranks using 1D Gaussian Mixture Model.

        Parameters:
        class_proportions (dict, optional): Proportions of each class.
        n_mix (int, optional): Number of mixtures to try.

        Returns:
        tuple: DataFrame with 1D Gaussian Mixture Model labels added and the maximum Kendall tau score.
        """
        if not(class_proportions is None):
            # Calculate weights for each cluster based on class proportions
            weights = np.array(list(class_proportions.values()))
            
            # Ensure weights are normalized according to scikit
            weights[-1] = 1-sum(weights[:-1])
        else:
            weights = None
            
        mixtures = []
        taumax = -1
        best_index = 0
        
        for i in range(n_mix): 
            mixtures.append(GaussianMixture(n_components = self.n_classes, weights_init = weights))
            
            mix_labels = mixtures[i].fit_predict(self.ranks)
        

            tau = kendalltau(self.ranks, mix_labels, variant = 'c').statistic

            if tau >= taumax:
                print(tau)
                taumax = tau
                best_index = i

        mixture_labels = pd.DataFrame(mixtures[best_index].predict(self.ranks), index = self.valid_indices)
        
        self.model = mixtures[best_index]
        
        full_scores = self.add_class(mixture_labels)
        return full_scores, taumax
    
    def get_predictions(self):
        """
        Gets the predictions from the clustering model.

        Returns:
        pd.DataFrame: DataFrame with clustering labels added.
        """
        labels = pd.DataFrame(self.model.predict(self.ranks), index = self.valid_indices)
        full_scores = self.add_class(labels)
        return(full_scores)

    def make_score(self, full_scores, n_classes = 7):
        """
        Creates a score based on the clustering labels.

        Parameters:
        full_scores (pd.DataFrame): DataFrame containing the full scores with labels.
        n_classes (int, optional): Number of classes for scoring.

        Returns:
        pd.DataFrame: DataFrame with ESG scores.
        """    
        mean_ranks = full_scores.groupby('labels').mean()
        global_mean_ranks = mean_ranks.mean(axis = 1)
        global_mean_ranks = global_mean_ranks.sort_values()
        
        sorted_labels = global_mean_ranks.index.tolist()  # Get the sorted cluster labels
        
        self.sorted_labels = sorted_labels
        # Create a mapping dictionary from original labels to sorted labels
        # Does a low value means a poor ESG score
        #label_mapping = {label: n_classes-rank for rank, label in enumerate(sorted_labels, start=1)}
        label_mapping = {label: rank for rank, label in enumerate(sorted_labels)}

        # Map the labels in the original DataFrame according to the sorted order
        full_scores['sorted_labels'] = full_scores['labels'].map(label_mapping)
        
        
        rank_stats = pd.DataFrame({'labels': global_mean_ranks.index, 'mean': global_mean_ranks.values})
        rank_stats['labels'].map(label_mapping)
        rank_stats['std'] = full_scores.groupby('sorted_labels').std().mean(axis=1)

        self.full_ranks = full_scores
        self.rank_stats = rank_stats

        ESGTV = pd.DataFrame({'Tag':self.valid_tickers,'Score': full_scores['sorted_labels']})

        ESGTV.dropna(inplace = True)
        return ESGTV
    
    def make_score_2(self, full_scores, n_classes = 7, gaussian = True):
        """
        Creates a score based on the clustering labels with additional statistics.

        Parameters:
        full_scores (pd.DataFrame): DataFrame containing the full scores with labels.
        n_classes (int, optional): Number of classes for scoring.
        gaussian (bool, optional): Whether to use Gaussian statistics.

        Returns:
        pd.DataFrame: DataFrame with ESG scores.
        """        
        scores = full_scores.copy()
        mean_ranks = scores.groupby('labels').mean()
        global_mean_ranks = mean_ranks.mean(axis = 1)
        global_mean_ranks = global_mean_ranks.sort_values()
        
        sorted_labels = global_mean_ranks.index.tolist()  # Get the sorted cluster labels
        
        self.sorted_labels = sorted_labels
        # Create a mapping dictionary from original labels to sorted labels
        # Does a low value means a poor ESG score
        #label_mapping = {label: n_classes-rank for rank, label in enumerate(sorted_labels, start=1)}
        label_mapping = {label: rank for rank, label in enumerate(sorted_labels)}

        # Map the labels in the original DataFrame according to the sorted order
        scores['sorted_labels'] = scores['labels'].map(label_mapping)
        
        rank_stats = pd.DataFrame({'labels': global_mean_ranks.index, 'mean': global_mean_ranks.values})
        rank_stats['labels'].map(label_mapping)
        rank_stats['mean_std'] = scores.groupby('sorted_labels').std().mean(axis=1)
        rank_stats['std_mean_rank'] = scores.groupby('sorted_labels').mean().std(axis = 1)
        
        if gaussian:
            covs = self.model.covariances_
            a = np.ones(4)/4
            clusters_std = []
            for k in self.sorted_labels:
                clusters_std.append(np.sqrt(a.T.dot(covs[k]).dot(a)))
            rank_stats['cluster_std'] = clusters_std

        self.full_ranks = scores
        self.rank_stats = rank_stats

        ESGTV = pd.DataFrame({'Tag':self.valid_tickers,'Score': scores['sorted_labels']})

        ESGTV.dropna(inplace = True)
        return ESGTV
    
    
    def quantiles_mixture(self, alpha = 0.95):
        """
        Calculates the quantiles for the mixture model.

        Parameters:
        alpha (float, optional): Quantile level.

        Returns:
        list: List of quantiles for each sample.
        """
        densities = self.model.predict_proba(self.ranks) # shape nsamples x ncomponents
        
        def goal(x, weights, means, stds):
            return sum(weights * norm.cdf(x, means, stds)) + alpha - 1
        
        def goal_deriv(x, weights, means, stds):
            return sum(weights * norm.pdf(x, means, stds))
        
        
        rank_stats = self.rank_stats.copy()
        means = rank_stats.sort_values(by = 'labels')["mean"]
        stds = rank_stats.sort_values(by = 'labels')["cluster_std"]
        
        roots = []
        for n, i in enumerate(self.full_ranks.index):

            weights = densities[n]
            
            func = lambda rank: goal(rank, weights, means, stds)
            deriv = lambda rank: [goal_deriv(rank, weights, means, stds)] # Need to return a list
            
            root = fsolve(func, x0 = means[self.full_ranks['sorted_labels'].loc[i]], fprime = deriv)
            roots += [root[0]] # Fsolve returns a singleton
        return roots
    
    def set_cluster_parameters(self):
        """
        Sets the cluster parameters such as densities, means, and standard deviations.
        """
        self.densities = self.model.predict_proba(self.ranks)
        rank_stats = self.rank_stats.copy()
        self.clusters_means = rank_stats.sort_values(by = 'labels')["mean"]
        self.clusters_stds = rank_stats.sort_values(by = 'labels')["cluster_std"]
        
    def get_cluster_parameters(self):
        """
        Gets the cluster parameters.

        Returns:
        tuple: Densities, cluster means, and cluster standard deviations.
        """
        return(self.densities, self.clusters_means, self.clusters_stds)
    
    def score_uncertainty(self, full_scores, eta = 1):
        """
        Calculates a manual uncertainty of the ESG scores.

        Parameters:
        full_scores (pd.DataFrame): DataFrame containing the full scores with labels.
        eta (float, optional): Scaling factor for the uncertainty. Default is 1.

        Returns:
        tuple: DataFrame with ESG scores, mean ranks, left and right uncertainty intervals.
        """
        scores = full_scores.copy()
        mean_ranks = scores.groupby('labels').mean()
        global_mean_ranks = mean_ranks.mean(axis = 1)
        global_mean_ranks = global_mean_ranks.sort_values()

        sorted_labels = global_mean_ranks.index.tolist()  # Get the sorted cluster labels
        
        self.sorted_labels = sorted_labels
        # Create a mapping dictionary from original labels to sorted labels
        # Does a low value means a poor ESG score
        #label_mapping = {label: n_classes-rank for rank, label in enumerate(sorted_labels, start=1)}
        label_mapping = {label: rank for rank, label in enumerate(sorted_labels)}
        reverse_mapping = {rank: label for rank, label in enumerate(sorted_labels)}

        # Map the labels in the original DataFrame according to the sorted order
        scores['sorted_labels'] = scores['labels'].map(label_mapping)
        
        rank_stats = pd.DataFrame({'labels': global_mean_ranks.index, 'mean': global_mean_ranks.values})
        #rank_stats['labels'].map(label_mapping)     

        covs = self.model.covariances_
        a = np.ones(4)/4
        clusters_std = []
        for k in self.sorted_labels:
            clusters_std.append(np.sqrt(a.T.dot(covs[k]).dot(a)))
        rank_stats['cluster_std'] = clusters_std

        self.full_ranks = scores
        self.rank_stats = rank_stats

        ESGTV = pd.DataFrame({'Tag':self.valid_tickers,'Score': scores['sorted_labels']})

        ESGTV.dropna(inplace = True)
        
        densities = self.model.predict_proba(self.ranks) # shape nsamples x ncomponents
        mean_ranks = []
        left_inc, right_inc = np.zeros(len(ESGTV.index)), np.zeros(len(ESGTV.index))
        for n, i in enumerate(ESGTV.index):
            count = 0
            left_incer, right_incer = 0, 0
            k_star = self.full_ranks['labels'].loc[i]
            cond = self.rank_stats["labels"] == k_star
            mean_rank = self.rank_stats[cond]["mean"].values[0]
            mean_ranks += [mean_rank]
            for k in self.sorted_labels:
                fac = densities[n][self.sorted_labels.index(k)] # reverse the mapping for k
                incer = eta * clusters_std[count] * fac
                low_rank, up_rank = max(1, mean_rank - incer), min(332, mean_rank + incer)
                #k_star = self.full_ranks['sorted_labels'].loc[i]
                if k < k_star :
                    left_incer += (mean_rank - low_rank)
                    if up_rank > mean_rank:
                        right_incer += (up_rank - mean_rank)
                    else:
                        left_incer += (mean_rank - up_rank)
                elif k == k_star:
                    left_incer += (mean_rank - low_rank)
                    right_incer += (up_rank - mean_rank)
                else:
                    right_incer += (up_rank - mean_rank)
                    if low_rank < mean_rank:
                        left_incer += (mean_rank - low_rank)
                    else:
                        right_incer += (low_rank - mean_rank)
                count += 1
            left_inc[n], right_inc[n] = left_incer, right_incer
        
        return ESGTV, mean_ranks, left_inc, right_inc
    
    def save_model(self,filename = 'model.pkl'):
        """
        Saves the clustering model to a file.

        Parameters:
        filename (str, optional): Filename to save the model. Default is 'model.pkl'.
        """
        with open(filename,'wb') as f:
            pickle.dump(self.model,f)


    def load_model(self,filename = 'model.pkl'):
        """
        Loads the clustering model from a file.

        Parameters:
        filename (str, optional): Filename to load the model from. Default is 'model.pkl'.
        """
        with open(filename, 'rb') as f:
           model = pickle.load(f)
           self.model = model
    
    def get_mean_ranks(self, labels_column = 'sorted_labels'):
        """
        Calculates the mean ranks for each cluster.

        Parameters:
        labels_column (str, optional): The column name for cluster labels. Default is 'sorted_labels'.

        Returns:
        pd.DataFrame: DataFrame containing mean ranks for each cluster.
        """
        agencies = self.full_ranks.columns[:4]
        rank_df = pd.DataFrame(columns = agencies)
        dfc = self.full_ranks.copy()
        for agency in agencies:
            mr = dfc[[agency, labels_column]].groupby(labels_column).mean().to_dict()
            rank_df[agency] = mr[agency]
            rank_df['cluster_mean_rank'] = rank_df.mean(axis=1)
        return rank_df

    def get_std_ranks(self, labels_column = 'sorted_labels'):
        """
        Calculates the standard deviation of ranks for each cluster.

        Parameters:
        labels_column (str, optional): The column name for cluster labels. Default is 'sorted_labels'.

        Returns:
        pd.DataFrame: DataFrame containing standard deviations of ranks for each cluster.
        """
        agencies = self.full_ranks.columns[:4]
        rank_df = pd.DataFrame(columns = agencies)
        dfc = self.full_ranks.copy()
        for agency in agencies:
            mr = dfc[[agency, labels_column]].groupby(labels_column).std().to_dict()
            rank_df[agency] = mr[agency]
            rank_df['cluster_std_rank'] = rank_df.mean(axis=1)
        return rank_df
    
    def plot_rank_uncer(self, tri_ranks, save = True, eng = True):
        """
        Plots the uncertainty of the ESG ranks.

        Parameters:
        tri_ranks (list): List of three ranks for the plot acting as a confidence interval (in the order min-max-mean).
        save (bool, optional): Whether to save the plot. Default is True.
        eng (bool, optional): Whether to use English labels. Default is True.
        """
        fig_size = (8,6)
        sns.set_theme()
        self.fig, self.ax = plt.subplots(figsize=fig_size)

        if eng:
            self.ax.plot(range(1,334), tri_ranks[2], 'bo', label = 'Mean')
            self.ax.fill_between(range(1,334), tri_ranks[0], tri_ranks[1], alpha = .25, color = 'g', label = 'Min-Max ESG')
                  
            self.ax.set_title('Uncertainty for each stock using the GMM')
            self.ax.set_xlabel('Index of the company')
            #self.ax.set_xlabel('Sorted number of companies')
            self.ax.set_ylabel('ESG')
        else:
            self.ax.plot(range(1,334), tri_ranks[2], 'bo', label = 'Rang moyen du cluster assigné')
            self.ax.fill_between(range(1,334), tri_ranks[0], tri_ranks[1], alpha = .25, color = 'g', label = "Rang minimal de l'entreprise")
                  
            self.ax.set_title('Incertitude du GMM pour chaque entreprise')
            self.ax.set_xlabel("Nombre d'entreprises")
            #self.ax.set_xlabel('Sorted number of companies')
            self.ax.set_ylabel('Rang ESG')
        
        self.ax.grid(True)
        
        self.ax.legend()
        plt.savefig("Figures/Incertitude_GMM.png")