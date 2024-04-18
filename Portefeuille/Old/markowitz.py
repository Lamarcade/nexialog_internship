# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:31:54 2024

@author: LoïcMARCADET
"""
import numpy as np
from numpy.linalg import inv
import secrets
import matplotlib.pyplot as plt

#%%

mu1,mu2 = 1.025,1.075
mu = np.array([mu1,mu2])
mu0 = 1.05
sigma1, sigma2 = 0.3, 0.5
s1s, s2s = 0.09, 0.25 #sigmas squared
cs12 = sigma1 * sigma2 * 0.5
sigma = np.ones((2,2)) 
sigma[0,:] = s1s, cs12
sigma[1,:] = cs12, s2s

R0 = 1 #risk-free asset return
c = 0.50 #solutions of investment problems coincide
V0 = 1

onevec = mu - R0 * np.ones(2,)
optimal_weights = V0/c * inv(sigma) @ onevec

# With the other investment problem
# ow = V0 *(mu0 - R0)* np.dot(inv(sigma), onevec) / (np.dot(np.dot(onevec, inv(sigma)), onevec))

w0 = 1- optimal_weights * np.ones(2,)

#%% Simulate mu and sigma

seed = secrets.randbits(128)  
rng = np.random.default_rng(seed)

opt_w = lambda mu, sigma: V0/c * inv(sigma) @ (mu - R0 * np.ones(2,))
opt_weights =  lambda w: np.insert(w,0, 1-sum(w))
hatsig = lambda w, sigma: np.sqrt(np.dot(np.dot(w,sigma),w))
hatmu = lambda w, mu: (1-sum(w))*R0 + np.dot(w,mu)

sample_weights = []
sighat, muhat = [],[]

for i in range(3000):
    samples = rng.multivariate_normal(mu, sigma, 200)

    muest = np.mean(samples, axis = 0)
    sigest = np.var(samples, axis = 0)
    varest = np.array([[sigest[0], 0.5 * sigest[0] * sigest[1]], [0.5 * sigest[0] * sigest[1], sigest[1]]])

    w = opt_w(muest,varest)
    sample_weights.append(opt_weights(w))
    sighat.append(hatsig(w,sigma))
    muhat.append(hatmu(w,mu))


#
conc = np.concatenate(sample_weights, axis=0)

# Reshape the concatenated array to have 3 columns
res = conc.reshape(-1, 3)

plt.plot(res[:,1], res[:,2], 'o')

plt.figure()
plt.plot(sighat,muhat, 'o')
