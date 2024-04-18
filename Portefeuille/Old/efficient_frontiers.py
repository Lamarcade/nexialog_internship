# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:01:42 2024

@author: LoïcMARCADET
"""

import numpy as np
from numpy.linalg import inv
import secrets
import matplotlib.pyplot as plt

#%%
R0 = 1.05 #risk-free asset return
c = 0.50 #solutions of investment problems coincide
V0 = 1

mu = np.array([1.05,1.15,1.1,1.1])
mu0 = 1.05
corrs = np.diag([0.3, 0.25, 0.2, 0.15])
std_list = [1, 0.6, 0.5, 0.4, 0.6, 1, -0.1, 0, 0.5, -0.1, 1, 0.6, 0.4, 0, 0.6, 1]

# Reshape the list into a 4x4 matrix
stds = np.array(std_list).reshape(4, 4)

sigma = np.dot(np.dot(corrs,stds),corrs)
vec4 = np.ones(4)
sigmainv = inv(sigma)
minvarweights = sigmainv @ vec4 / (vec4 @ sigmainv @ vec4)


#onevec = mu - R0 * np.ones(2,)
#optimal_weights = V0/c * inv(sigma) @ onevec

def no_risk_free(c, sigma = sigma, mu = mu):
    coeff = V0 / c 
    n = len(mu)
    on = np.ones(n)
    sigmainv = inv(sigma)
    fund = on @ sigmainv
    vec = sigmainv @ (mu - (max(fund@mu - c,0) / (fund@on)) * on)
    return coeff*vec

def with_risk_free(c, sigma = sigma, mu = mu):
    coeff = V0 / c 
    n = len(mu)
    on = np.ones(n)
    
    vec = inv(sigma) @ (mu - R0 * on)
    return coeff*vec


getsig = lambda w, sigma: np.sqrt(w@sigma@w)
getmu = lambda w, mu: w@mu
getmurf = lambda w, mu, V0: w@mu + (1-sum(w))*R0

#%%

cs = np.arange(1,1000)
sigmas, mus = [],[]
sigmas2, mus2 = [], []

for c in cs:
    w = no_risk_free(c)
    sigmas.append(getsig(w,sigma))
    mus.append(getmu(w,mu))
    w2 = with_risk_free(c)
    sigmas2.append(getsig(w2,sigma))
    mus2.append(getmurf(w2,mu,V0))    

minvarsig = getsig(minvarweights, sigma)
minvarmu = getmu(minvarweights, mu)

tangency = with_risk_free(vec4@sigmainv@(mu-R0*vec4))
sigt, mut = getsig(tangency, sigma), getmu(tangency, mu)

plt.figure()
plt.plot(sigmas,mus)
plt.plot(sigmas2,mus2, ls = 'dotted', color = 'g')
plt.plot(minvarsig, minvarmu, marker='+', color='r', markersize=10, label = "Minimum Variance portfolio")
plt.plot(sigt, mut, marker='o', color='r', markersize=5, label = "Tangent Portfolio")
plt.ylim(1,1.3)
plt.xlim(0,0.25)
plt.title('Efficient frontier with and without risk-free asset')
plt.legend()

plt.figure()
plt.plot(sigmas,mus)
plt.plot(sigmas2,mus2, ls = 'dotted', color = 'g')
plt.plot(minvarsig, minvarmu, marker='+', color='r', markersize=10)
plt.plot(sigt, mut, marker='o', color='r', markersize=5)
plt.ylim(1.135, 1.17)
plt.xlim(0.105,0.130)