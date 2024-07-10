# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:21:31 2024

@author: LoïcMARCADET
"""

from Tester import Tester

#%%
Te = Tester()
Te.set_scores()

#%%
#Te.sector_evolution(130, 320, 30)

#%%
Te.set_agencies_df_list(ranks = False, scores = True)
Te.stocks_init(3)
#Te.esg_frontiers(80, 92, 2, 3)
#Te.sector_esg_frontiers(3)
#%%
Te.set_agencies_df_list()
Te.stocks_init(3)
#Te.esg_frontiers(130, 320, 30, 3)

#%%
Te.set_agencies_df_list(ranks = False, scores = False, standard=True)
Te.clusters_variables(standardize = True)
Te.harmonized_variables()
#Te.combine_TV_lists()
#Te.sharpe_analysis()

#%%
Te.set_agencies_df_list()
Te.sharpe_exclusion()