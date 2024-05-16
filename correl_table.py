# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:03:56 2024

@author: LoïcMARCADET
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from ScoreGetter import ScoreGetter

SG = ScoreGetter('ESG/Scores/')
SG.reduced_df()
scores_df = SG.keep_valid()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

SG.valid_ticker_sector()

sectors_list = SG.get_valid_sector_df()
correls = np.zeros((4,4))

agency_order = ['MS', 'SU', 'SP', 'RE']
names = ['MSCI', 'Sust.', 'S&P', 'Refi.']
for i in range(4):
    correls[i][i] = 1
    for j in range(4):
        if i!= j:
            correls[i][j], _ = kendalltau(scores_df[agency_order[i]],scores_df[agency_order[j]], variant = 'b')

mask = np.triu(np.ones_like(correls, dtype=bool))

plt.figure()
sns.heatmap(correls, annot = True, mask = mask, xticklabels=names, yticklabels = names, fmt = ".2f", cmap = 'viridis')
fr = False
if fr:
    plt.title('Corrélation de rang des scores ESG \n pour les entreprises du S&P 500')
    plt.savefig('Figures/correlations_fr.png')
else:
    plt.title('Rank correlation of ESG scores \n for companies in the S&P 500')
    plt.savefig('Figures/correlations.png')    

