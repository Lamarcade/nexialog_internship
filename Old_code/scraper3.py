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

# create webdriver object
driver = webdriver.Edge()

# get companies tickers 
tags = []
with open('tickers.csv', 'r') as tk:
    csvFile = csv.reader(tk)
    for lines in csvFile:
        tags += lines

#Remove column name
tags.pop(0)

#%% Get ESG scores

#tags = ['A','AAL'] # Test

scores = []

# reject all button counter
count = 0

for tag in tags: 

    # get yahoo finance.com
    is_link ='https://finance.yahoo.com/quote/' + tag + "/sustainability"
    driver.get(is_link)
    time.sleep(3)
    
    # click Reject all the first time on the website
    if count == 0:
        # get element 
        RejectAll= driver.find_element(By.XPATH, '//button[@class="btn secondary reject-all"]')
        # create action chain object
        action = ActionChains(driver)
        # click the item
        action.click(on_element = RejectAll)
        # perform the operation
        action.perform()
        
        count += 1
        time.sleep(3)

    # check if the ticker was found
    current_page = driver.current_url
    #print(current_page)
    #print(is_link)
    if not(is_link in current_page):
        print(tag)
        print()
        
        # add NA if the company is not present
        scores.append('NA')
    else:
        # check if data is available
        availability = driver.find_element(By.ID, "Col1-0-Sustainability-Proxy").text
        if ("not available") in availability:
            scores.append('NA')
        else:
            # XPath of the ESG Score
            # cl = "Fz(36px) Fw(600) D(ib) Mend(5px)"
            score = driver.find_element(By.XPATH, "//div[contains(@class, 'Fz(36px) Fw(600) D(ib) Mend(5px)')]").text
            #print(score)
            scores.append(score)
        
print(scores)

#%%
tags10 = tags[:10]

with open('scores.csv', 'w', newline='') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=' ')
    for i, (tag, score) in enumerate(zip(tags, scores)):
        scorewriter.writerow([tag] + [score])

