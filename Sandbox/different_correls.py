# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:16:13 2024

@author: LoïcMARCADET
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

sns.set_theme()

#X = [4,2,6,2,5,9, 3, 7, 8]
Y = [40,30,40,35,45,70, 40, 60,60]
X = np.arange(-1,1.05, 0.1)
Z = np.exp(5*X) 
Z /= np.std(Z)

plt.figure()
plt.plot(X,Z, 'bo')
plt.xlabel('X1')
plt.ylabel('Y1')


#P, S, TB, TC = pearsonr(X,Y), spearmanr(X,Y), kendalltau(X,Y), kendalltau(X,Y, variant = 'c')
PZ, SZ, TBZ, TCZ = pearsonr(X,Z), spearmanr(X,Z), kendalltau(X,Z), kendalltau(X,Z, variant = 'c')

A = [8,8,8,8,8, 8, 8, 8, 19]
B = [5,5.5,6, 6.5, 7, 8,8.5,9, 13]

A = (A - np.mean(A)) / np.std(A)
B = (B- np.mean(B)) / np.std(B)

PA, SA, TBA, TCA = pearsonr(A,B), spearmanr(A,B), kendalltau(A,B), kendalltau(A,B, variant = 'c')

plt.figure()
plt.plot(A,B, 'bo')
plt.xlabel('X2')
plt.ylabel('Y2')