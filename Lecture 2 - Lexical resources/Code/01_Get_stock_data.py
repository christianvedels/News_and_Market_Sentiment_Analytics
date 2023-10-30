# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:28:58 2023

@author: chris
"""
#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Parameters
end_date = datetime.now().strftime('%Y-%m-%d') # Today
start_date = '2023-10-01'
interval = '1h'

# %% Getting stock market data
# Novo Nordisk
nvo = yf.Ticker("NVO")

nvo_hist = nvo.history(start=start_date, end=end_date, interval=interval)
print(nvo_hist)

# %% Plotting price
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

sp500_hist = sp500.history(start=start_date, end=end_date, interval=interval)
print(sp500_hist)

sp500_hist.to_csv("SP500_index.csv")

# %% Get news
from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key="INSERT KEY")

# /v2/top-headlines
news = newsapi.get_everything(
    q='novo nordisk',
    from_param=start_date,
    to=end_date,
    language='en'
    )

articles = news["articles"]
articles[0]

# %% Structure data
news_data = []
for news_item in articles:
    news_time = datetime.fromisoformat(news_item['publishedAt']).strftime('%Y-%m-%d %H:%M:%S')
    news_source = news_item['source']['name']
    news_headline = news_item['title']
    news_content = news_item['content']
    news_data.append([news_time, news_source, news_headline, news_content])
    
# Create a DataFrame from the news data
news_df = pd.DataFrame(news_data, columns=['Date', 'Source', 'Headline', 'Content'])

# Save csv
news_df.to_csv("Novo_Nordisk_News.csv")
