# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:28:58 2023

@author: chris
"""

# %% Libraries
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Parameters
end_date = datetime.now().strftime('%Y-%m-%d') # Today
start_date = '2022-01-01'

# %% Getting stock market data
# Novo Nordisk
nvo = yf.Ticker("NVO")

nvo_hist = nvo.history(start=start_date,end=end_date)
print(nvo_hist)

# %% Plotting price
# Create a DataFrame for the sentiment scores
plt.figure(figsize=(12, 6))
sns.lineplot(data=nvo_hist, x=nvo_hist.index, y='Close', label='Closing price', color='b')
plt.xlabel('Date')
plt.title('Novo Nordisk closing price')
plt.legend()
plt.show()

# %% Save Novo Nordisk
nvo_hist.to_csv("Novo_Nordisk_prices.csv")

# %% Getting S&P 500 index and saving it
sp500 = yf.Ticker("^GSPC")

sp500_hist = sp500.history(start=start_date,end=end_date)
print(sp500_hist)

sp500_hist.to_csv("SP500_index.csv")

# %% Getting news
nvo_news = nvo.get_news()

# %% Extract date and headlines
news_data = []
for news_item in nvo_news:
    news_time = datetime.utcfromtimestamp(news_item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')
    news_headline = news_item['title']
    news_data.append([news_time, news_headline])

# Create a DataFrame from the news data
news_df = pd.DataFrame(news_data, columns=['Date', 'Headlines'])

# Save csv
news_df.to_csv("Novo_Nordisk_News.csv")
