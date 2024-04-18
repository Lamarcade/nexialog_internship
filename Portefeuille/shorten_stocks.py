# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:59:56 2024

@author: LoïcMARCADET
"""

import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('sp500_stocks.csv')

# Remove the specified column from the DataFrame
df = df.drop('High', axis=1)
df = df.drop('Low', axis=1)
df = df.drop('Close', axis=1)
df = df.drop('Open', axis=1)
df = df.drop('Volume', axis=1)

# Write the updated DataFrame to a new CSV file
df.to_csv('sp500_stocks_short.csv', index=False)