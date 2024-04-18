# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:21:40 2024

@author: LoïcMARCADET
"""

# import yfinance, pandas and os
import yfinance as yf
import pandas as pd
import os

cola = "KO"
cola_y = yf.Ticker(cola)
esg_data = pd.DataFrame.transpose(cola_y.sustainability)
esg_data['company_ticker'] = str(cola_y.ticker)

# Import list of tickers from file
#os.chdir("C:\...")
djia = pd.read_csv('tickers.csv')
# Retrieve Yahoo! Finance Sustainability Scores for each ticker
for i in djia['Symbol']:
    # print(i)
    i_y = yf.Ticker(i)
    try:
        if i_y.sustainability is not None:
            temp = pd.DataFrame.transpose(i_y.sustainability)
            temp['company_ticker'] = str(i_y.ticker)
            # print(temp)
            esg_data = esg_data.append(temp)
    except IndexError:
        pass
    
esg_data.to_csv('djia_sustainability_scores.csv', encoding='utf-8')

