# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:46:31 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ESG_Portfolio import ESG_Portfolio
from Stocks import Stocks
import copulas
from scipy.stats import norm, kendalltau
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import GaussianUnivariate
from copulas.visualization import compare_3d
from plotly.offline import plot
import plotly.graph_objs as go

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

sectors_list = SG.get_valid_sector_df()

#%%

plt.figure()
s = sns.pairplot(scores_ranks[['MS', 'SU','SP','RE']], corner = True)
s.fig.suptitle('Paires de rang des agences', y = 1.03)
#plt.savefig("Figures/rangs.png", bbox_inches = 'tight')

#%% Load trained model

load_GSM = ScoreMaker(scores_ranks, dict_agencies, valid_tickers, valid_indices, 7)
load_GSM.load_model('gauss.pkl')

full_scores = load_GSM.get_predictions()

ESGTV5, mean_ranks, left_inc, right_inc = load_GSM.score_uncertainty(full_scores, eta = 1)

load_GSM.set_cluster_parameters()
densities, means, stds = load_GSM.get_cluster_parameters()

true_mean_ranks = scores_ranks.mean(axis = 1)

def cdf_sum(x, weights, means, stds):
    return sum(weights * norm.cdf(x, means, stds))

n, k = densities.shape
rank_transfo = np.zeros(n)
for i in range(n):
    #f = lambda x: cdf_sum(x, densities[i], means, stds)
    rank_transfo[i] = cdf_sum(true_mean_ranks.values[i], densities[i], means, stds)

#%%
taus = np.zeros((4,4))

agency_order = ['MS', 'SU', 'SP', 'RE']
names = ['MSCI', 'Sust.', 'S&P', 'Refi.']
for i in range(4):
    taus[i][i] = 1
    for j in range(4):
        if i!= j:
            taus[i][j], _ = kendalltau(scores_ranks[agency_order[i]],scores_ranks[agency_order[j]], variant = 'b')

rhos = np.zeros((4,4))      
for i in range(4):
    for j in range(4):
        rhos[i][j] = np.sin(np.pi / 2 * taus[i][j])
#%%
rank_df = pd.DataFrame({'Tag': valid_tickers, "Score": true_mean_ranks})

#%% 
copula = GaussianMultivariate()
copula.fit(scores_ranks) 

#%%
num_samples = 503
synthetic_data = copula.sample(num_samples)
synthetic_data.head()


fig = compare_3d(scores_ranks[["MS", "SU", "SP"]], synthetic_data[["MS", "SU", "SP"]])
plot(fig, auto_open = True)


#%% Get the stock data and keep the companies in common with the target variable
st = Stocks(path, annual_rf)
st.process_data()
st.compute_monthly_returns()

# 0: MSCI 1: Sustainalytics 2: S&P 3: Refinitiv
provider = 'Mean Ranks'
_ = st.keep_common_tickers(rank_df, sectors_list)
#_ = st.keep_common_tickers(ESGTV, sectors_list)

stocks_sectors, stocks_ESG = st.select_assets(5)
#stocks_ESG = st.restrict_assets(50)
st.compute_mean()
st.compute_covariance()
mean, old_cov , rf = st.get_mean(), st.get_covariance(), st.get_rf()
cov = st.covariance_approximation()

#st.plot_sectors()

#%% Build a portfolio with restrictions on the minimal ESG score

epf = ESG_Portfolio(mean,cov,rf, stocks_ESG, short_sales = False)

constraint = 250
weights = epf.optimal_portfolio_ESG(constraint, input_bounds = None)

n_tests = 100
min_rank = 230
count = 0
for i in range(n_tests):
    synthetic_data = copula.sample(num_samples)
    synthetic_mean_ranks = synthetic_data.mean(axis = 1)
    epf.set_ESGs(synthetic_mean_ranks[stocks_sectors.index])
    esg = epf.get_ESG(weights)
    if esg <= min_rank:
        count +=1
    
