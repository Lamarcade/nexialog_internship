# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:30:57 2024

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

#%% Utilitary functions

def clean_info(ticker,sector):
    cleaned_ticker = ticker.strip('()')
    cleaned_sector = sector.replace("Industry: ", "")
    return cleaned_ticker, cleaned_sector

def extract_scores(input_string):
    parts = input_string.split('\n')
    last_date_index = -1

    # Find the index of the last date in the string
    # Dates are in format 'Apr-23'
    for i, part in enumerate(parts):
        if '-' in part:
            last_date_index = i

    # If no date is found, return the original string
    if last_date_index == -1:
        return parts

    # Extract data up to the last date
    extracted_parts = parts[:last_date_index + 1]

    return extracted_parts

def split_scores_dates(scores):
    if (len(scores) == 10):
        return(scores[:5], scores[5:10])
    else:
        return(None)


#%% 

# create webdriver object
driver = webdriver.Edge()

#%% 

# Go on the MSCI ESG Ratings search tool
is_link ='https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool/'
driver.get(is_link)
time.sleep(3)

# Reject all cookies
RejectAll= driver.find_element(By.ID, 'onetrust-reject-all-handler')
# create action chain object
action = ActionChains(driver)
# click the item
action.click(on_element = RejectAll)
# perform the operation
action.perform()
time.sleep(3)

#%% Get companies tickers 
tags = []
with open('tickers.csv', 'r') as tk:
    csvFile = csv.reader(tk)
    for lines in csvFile:
        tags += lines

#Remove column name
tags.pop(0)

#%%

#Set up variables
company_tickers, company_names, company_sectors, company_scores, company_dates = [],[],[],[],[]
tagslim = tags[:500]

for tag in tags:
    
    ### Find the company 
    
    #
    SearchBar = driver.find_element(By.XPATH, "//input[contains(@placeholder,'Search by company or ticker')]")
        
    # Only on certain OS, reset last input
    SearchBar.send_keys(Keys.CONTROL + "a")
    SearchBar.send_keys(Keys.DELETE)

    SearchBar.send_keys(tag)
    time.sleep(3)

    SearchResult = driver.find_element(By.ID, 'ui-id-1')
    
    try:
        #SearchResult.click()
        asb = ActionChains(driver)
        asb.click(on_element = SearchResult)
        asb.perform()
        time.sleep(3)
        
        ###  Retrieve data
        
        Toggle = driver.find_element(By.ID, 'esg-transparency-toggle-link')
        t_action = ActionChains(driver)
        t_action.click(on_element = Toggle)
        t_action.perform()
        time.sleep(3)
        
        company_name = driver.find_element(By.XPATH, "//h1[contains(@class, 'header-company-title')]").text
        ticker = driver.find_element(By.XPATH, "//div[contains(@class, 'header-company-ticker')]").text
        sector = driver.find_element(By.XPATH, "//div[contains(@class, 'header-esg-industry')]").text
        
        company_ticker, company_sector = clean_info(ticker,sector)
        
        
        scores = driver.find_element(By.ID, "_esgratingsprofile_esg-rating-history")
        list_scores = scores.text
        
        scores_dates = extract_scores(list_scores)
            
        company_score, company_date = split_scores_dates(scores_dates)
        
    except:
        company_ticker, company_name, company_sector = tag,'NAN','NAN'
        company_score = ['NAN','NAN','NAN','NAN','NAN']
        company_date = ['NAN','NAN','NAN','NAN','NAN']
    
    company_tickers.append(company_ticker)
    company_names.append(company_name)
    company_sectors.append(company_sector)
    company_scores.append(company_score)
    company_dates.append(company_date)

#%%

#tags10 = tags[:10]

with open('MSCI_scores.csv', 'w', newline='') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=' ')
    for (ticker, name, sector, scores, dates) in zip(company_tickers, company_names, company_sectors, company_scores, company_dates):
        aggreg = [ticker] + [name] + [sector] + scores + dates
        scorewriter.writerow(aggreg)


#%%
# =============================================================================
#     # click Reject all the first time on the website
#     if count == 0:
#         # get element 
#         RejectAll= driver.find_element(By.XPATH, '//button[@class="btn secondary reject-all"]')
#         # create action chain object
#         action = ActionChains(driver)
#         # click the item
#         action.click(on_element = RejectAll)
#         # perform the operation
#         action.perform()
#         
#         count += 1
#         time.sleep(3)
# =============================================================================

# =============================================================================
#     # check if the ticker was found
#     current_page = driver.current_url
#     #print(current_page)
#     #print(is_link)
#     if not(is_link in current_page):
#         print(tag)
#         print()
#         
#         # add NA if the company is not present
#         scores.append('NA')
#     else:
#         # check if data is available
#         availability = driver.find_element(By.ID, "Col1-0-Sustainability-Proxy").text
#         if ("not available") in availability:
#             scores.append('NA')
#         else:
#             # XPath of the ESG Score
#             # cl = "Fz(36px) Fw(600) D(ib) Mend(5px)"
#             score = driver.find_element(By.XPATH, "//i[contains(@class, 'msci-icon-search')]").text
#             #print(score)
#             scores.append(score)
#         
# print(scores)
# =============================================================================
