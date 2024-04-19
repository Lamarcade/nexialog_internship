# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:36:23 2024

@author: LoïcMARCADET
"""

#%% Libraries 
import matplotlib.pyplot as plt  
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import random as rd

from scores_utils import *

plt.close('all') 

#%% Retrieve dataframes

MS, SU, SP, RE = get_scores()
MSS, SUS, SPS, RES = reduced_df(MS, SU, SP, RE)

# All the agencies
dict_agencies = {'MSCI': MSS['Score'], 'SU': SUS['Score'], 'SP' : SPS['Score'], 'RE': RES['Score']}
scores = get_score_df(dict_agencies)
scores_valid, valid_indices = keep_valid(scores)
std_scores = standardise_df(scores)
scores_ranks = scores_valid.copy()
scores_ranks = scores_ranks.rank()


#triplet = std_scores[:,1:4]

#triplet_df = pd.DataFrame(triplet, columns = dict_agencies.keys())

#%% 
scale = np.arange(100)
MSv = scores_valid['MSCI']
SPv = scores_valid['SP']
REv = scores_valid['RE']

def mapping(score, params):
    letters = ['CCC','B', 'BB', 'BBB','A','AA','AAA']
    scores = list(range(7))
    
    for i in range(len(params)):
        if score <= params[i]:
            return(scores[i])
    return 6


def scale_correl(params, MSv,SPv,REv):
    SP2, RE2 = SPv.copy(), REv.copy()
    
    #SP2['Score'] = SP2['Score'].apply(lambda x: mapping(x, params))
    #RE2['Score'] = RE2['Score'].apply(lambda x: mapping(x, params))
    SP2 = SP2.apply(lambda x: mapping(x, params))
    RE2 = RE2.apply(lambda x: mapping(x, params))
    
    #MS2 = MSS.iloc[:-1]
    #SP2 = SP2.iloc[:-1]
    tau1, _ = kendalltau(MSv, SP2)
    tau2, _ = kendalltau(MSv, RE2)
    
    return (tau1 + tau2) / 2

def optimize_scale(MSS,SPS,RES):
    objective = lambda params: -scale_correl(params, MSS,SPS,RES)
    init_guess = [10 + int(100 * i / 7) for i in range(7)]
    bounds = [(0.0, 100.0) for _ in range(7)]

    result = minimize(objective, init_guess, method='trust-constr', bounds=bounds)
    
    return result



def individual(length, low = 2, up = 100):
    ind = []
    tmp = np.random.randint(low,up)
    for i in range(length):
        while (tmp in ind):
            tmp = np.random.randint(low,up)
        ind.append(tmp)
        
    return np.sort(ind)

def mutate(ind, index, low = 2, up = 100):
    if index == 0:
        return np.random.randint(low, ind[index+1])
    elif index == (len(ind)-1):
        return np.random.randint(ind[index-1], up)
    else:
        return np.random.randint(ind[index-1],ind[index+1])

def population(count, length, low = 2, up = 100):
    return [individual(length, low, up) for _ in range(count)]

def fitness(individual, MSv, SPv, REv):
    res = scale_correl(individual, MSv,SPv,REv)
    return(res)

def mean_fitness(popu, MSv, SPv, REv):
    res = 0
    for ind in popu:
        res += fitness(ind, MSv, SPv, REv)
    return(res/len(popu))

def sort_best(popu, cache, MSv, SPv, REv):
    sorted_pop = []
    for ind in popu:
        h = hash(tuple(ind))
        if h in cache:
            fit = cache[h]
        else:
            fit = fitness(ind, MSv, SPv, REv)
            cache[h] = fit

        sorted_pop.append( (ind, fit) )
    return sorted(sorted_pop, key=lambda t: -t[1])

def evolve(popu, cache, MSv, SPv, REv, low = 2, up = 100, mutate_chance = 0.1, best_kept_prop = 0.4, rd_kept_chance = 0.1):
    wanted_count = len(popu)
    pop_fit = sort_best(popu, cache, MSv, SPv, REv)
    kept_count = int(np.floor(best_kept_prop*len(popu)))
    avg_fit = 0

    spop = []

    #Compute average fitness
    for ind, fit in pop_fit :
        avg_fit += fit
        spop.append(ind)
    avg_fit /= wanted_count

    next_gen = spop[:kept_count]
    ind_len = len(popu[0])
    given_len = int(np.floor(ind_len/2))

    #Keep random individuals to avoid local maxima
    for ind in spop[kept_count:]:
        if rd.random() < rd_kept_chance :
            next_gen.append(ind)
    
    #Breed individuals
    
    wanted_children = wanted_count - len(next_gen)
    children_count = 0
    children = []
    while children_count < wanted_children:
        couple = rd.sample(next_gen, 2)
        child = np.concatenate((couple[0][:given_len],couple[1][given_len:]))
        if np.sort(child) == child:
            children.append(child)
            children_count +=1
    next_gen.extend(children)

    #Mutate some individuals to diversify pool
    for ind in next_gen :
        if rd.random() < mutate_chance :
            index = rd.randint(0, ind_len-1)
            ind[index] = mutate(ind,index,low,up)

    return(next_gen, avg_fit)

def genetics(count, length, max_gen, accuracy, MSv, SPv, REv, low = 2, up = 100, mutate_chance = 0.1, best_kept_prop = 0.5, rd_kept_chance = 0.1):
    popu = population(count, length, low, up)
    gen_count = 0
    avg_fit = accuracy + 1 

    known_fit = {}

    while (gen_count < max_gen) and (avg_fit < accuracy) :
        gen_count += 1
        popu, avg_fit = evolve(popu, known_fit, MSv, SPv, REv, low, up, mutate_chance, best_kept_prop, rd_kept_chance)
        print()
        print("Precedent average fitness was", avg_fit)
    return(sort_best(popu, known_fit, MSv, SPv, REv))