# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:24:15 2024

@author: LoïcMARCADET
"""

import numpy as np
from numpy.linalg import inv
import secrets
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

#%%
R0 = 1.05 #risk-free asset return
cst = 0.50 #solutions of investment problems coincide
V0 = 10

mu = np.array([1.05,1.15,1.1,1.1])
mu0 = 1.05
corrs = np.diag([0.3, 0.25, 0.2, 0.15])
std_list = [1, 0.6, 0.5, 0.4, 0.6, 1, -0.1, 0, 0.5, -0.1, 1, 0.6, 0.4, 0, 0.6, 1]

# Reshape the list into a 4x4 matrix
stds = np.array(std_list).reshape(4, 4)

#%% 

#%%
getsig = lambda w, sigma: np.sqrt(w@sigma@w)
getmu = lambda w, mu: w@mu
getmurf = lambda w, mu, V0: w@mu + (1-sum(w))*R0
#%% 
def efficient_frontier(cs, mu = mu, stds = stds, V0 = V0):
    n = len(mu)
    cs = np.arange(1,1000)
    sigmas, mus = [],[]

    for c in cs:
        mean_variance = lambda w:(c/(2*V0) * w@stds@w - w @ mu)
        
        bounds = tuple((0, 1) for _ in range(n))
        constraint = LinearConstraint(np.ones(n), ub = V0)

        res_weights = minimize(mean_variance, np.ones(n)/n, bounds=bounds, constraints=constraint).x
        sigmas.append(getsig(res_weights,stds))
        mus.append(getmu(res_weights,mu))
        
    return sigmas, mus

#%%

sigmas, mus = efficient_frontier(np.arange(0.5,1000))

plt.figure()
plt.plot(sigmas,mus, 'o')

#plt.ylim(1,1.3)
#plt.xlim(0,0.25)
plt.title('Efficient frontier without risk-free asset')
plt.legend()



