# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:23:18 2024

@author: LoïcMARCADET
"""

import csv
from collections import defaultdict

#%% 
lei = defaultdict(list)

with open('entre.csv', 'r') as tk:
    csvFile = csv.reader(tk, delimiter = ';')

    for row in csvFile:
        head, *tail = row
        lei[head].append(tail)


#%% Remove column name
#lei.pop(0)
#names.pop(0)
#%%
conv = defaultdict(list)
with open('lei_ric_mapping.csv', 'r') as li:
    csvFile = csv.reader(li, delimiter = ';')
    count = 0
    for row in csvFile:
       head, *tail = row
       conv[head].append(tail)
       
#%%
# conv : (LEI) -> RIC
# lei : (LEI) -> Name

correspondence = {}

for lei_key, _ in lei.items():
    if lei_key in conv:
        ric_list = conv[lei_key]
        # Check if ISIN list is not empty
        if ric_list:
            # Note : Only the first ISIN of possibly many
            first_ric = ric_list[0]
            # Store the correspondence
            correspondence[lei_key] = first_ric
            
#%%

with open('LEI2RIC.csv', 'w', newline='') as csvfile:
    leiwriter = csv.writer(csvfile, delimiter=' ')
    for lei, ric  in correspondence.items():
        leiwriter.writerow([lei] + ric)
