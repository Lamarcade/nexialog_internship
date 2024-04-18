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
with open('lei-isin.csv', 'r') as li:
    csvFile = csv.reader(li, delimiter = ',')
    count = 0
    for row in csvFile:
       head, *tail = row
       conv[head].append(tail)
       
#%%
# conv : (Lei) -> ISIN
# lei : (LEI) -> Name

correspondence = {}

for lei_key, _ in lei.items():
    if lei_key in conv:
        isin_list = conv[lei_key]
        # Check if ISIN list is not empty
        if isin_list:
            # Note : Only the first ISIN of possibly many
            first_isin = isin_list[0]
            # Store the correspondence
            correspondence[lei_key] = first_isin
            
#%%

with open('LEI2ISIN.csv', 'w', newline='') as csvfile:
    leiwriter = csv.writer(csvfile, delimiter=' ')
    for lei, isin  in correspondence.items():
        leiwriter.writerow([lei] + isin)
