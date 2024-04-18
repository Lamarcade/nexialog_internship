# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:34:45 2024

@author: LoïcMARCADET
"""

import time
import csv
# import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
# import Action chains 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import re

# get companies tickers 
names = []
with open('company_names.csv', 'r') as tk:
    csvFile = csv.reader(tk)
    for lines in csvFile:
        names += lines

def remove_suffixes(company_names):
    name_extras = ['ltd', 'inc', 'corp', 'co', 'plc', 'Cl', 'A', 'Na', 'S', 'B']
    suffix_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(extra) for extra in name_extras) + r')\b', re.IGNORECASE)

    cleaned_names = [re.sub(suffix_pattern, '', name).strip() for name in company_names]

    return cleaned_names

name_extras = ['ltd', 'inc', 'corp', 'co', 'plc', 'Cl','A', 'Na', 'S']

#Remove column name
names.pop(0)
names = remove_suffixes(names)

tags = []
with open('tickers.csv', 'r') as tk:
    csvFile = csv.reader(tk)
    for lines in csvFile:
        tags += lines

#Remove column name
tags.pop(0)


#%% Get ESG scores

# create webdriver object
driver = webdriver.Edge()

names10 = names[:10] # Test


# reject all button counter
count = 0

# get yahoo finance.com
is_link ='https://www.spglobal.com/esg/scores/results?cid=4023623'
old_url = is_link
driver.get(is_link)
time.sleep(3)

#%%

scores = []
industries = []

# Test lock content
# names = ['AIMS', 'Apple']

for name in names: 

    SearchBar = driver.find_element(By.XPATH, "//input[contains(@placeholder,'Find a company')]")
    
    # Only on certain OS
    SearchBar.send_keys(Keys.CONTROL + "a")
    SearchBar.send_keys(Keys.DELETE)
    #
    SearchBar.send_keys(name)
    SearchBar.send_keys(Keys.SPACE)
    SearchBar.send_keys(Keys.ENTER)
    time.sleep(5)
    # check if data is available
    #availability = driver.find_element(By.ID, "Col1-0-Sustainability-Proxy").text

    current_url = driver.current_url
    if not (old_url in current_url):
        try:
            locked = driver.find_element(By.XPATH, "//p[contains(@class, 'lock__content')]").text
            score = 'NA'
            cname = name
            ticker = 'NA'
            industry = 'NA'
        except:
            score = driver.find_element(By.XPATH, "//p[contains(@class, 'scoreModule__score')]").text
            cname = driver.find_element(By.ID, "company-name").text
            industry = driver.find_element(By.ID, "company-industry").text

    else:
        score = 'NA'
        cname = name
        ticker = 'NA'
        industry = 'NA'
    #print(score)
    old_url = current_url
    scores.append(score)
    industries.append(industry)
        
print(scores)
#%%
# =============================================================================
# industries10 =  ['Industry: LIF Life Sciences Tools & Services',
#  'Industry: AIR Airlines',
#  'Industry: RTS Retailing',
#  'Industry: REI Equity Real Estate Investment Trusts (REITs)',
#  'Industry: BTC Biotechnology',
#  'NA',
#  'Industry: MTC Health Care Equipment & Supplies',
#  'Industry: INS Insurance',
#  'Industry: TSV IT services',
#  'Industry: SOF Software']
# 
# =============================================================================
def remove_industry(industries, chain):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(extra) for extra in chain) + r')\b', re.IGNORECASE)

    cleaned_names = [re.sub(pattern, '', ind).strip() for ind in industries]
    cleaned_names = [re.sub(': ', '', ind) for ind in cleaned_names]
    return cleaned_names

industries = remove_industry(industries, ['Industry']) 
#industries10 = remove_industry(industries10, [': '])

#%%

with open('scores.csv', 'w', newline='') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=' ')
    scorewriter.writerow(['Name'] + ['Tag'] + ['Score'] + ['Industry'])
    for i, (name, tag, score, industry) in enumerate(zip(names, tags, scores, industries)):
        print(name, industry)
        scorewriter.writerow([name] + [tag] + [score] + [industry])

