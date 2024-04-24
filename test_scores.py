# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:22:45 2024

@author: LoïcMARCADET
"""

from ScoreGetter import *
from ScoreMaker import *

SG = ScoreGetter('ESG/Scores/')
SG.reduced_mixed_df()
score_ranks = SG.get_rank_df()
dict_agencies = SG.get_dict()
valid_tickers, valid_indices = SG.get_valid_tickers(), SG.get_valid_indices()

SM = ScoreMaker(score_ranks, dict_agencies, valid_tickers, valid_indices, 7)

SMK = SM.kmeans()

ESGTV = SM.make_score(SMK)