# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:45:11 2023

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
import numpy as np

# %% Parameters
end_date = datetime.now().strftime('%Y-%m-%d') # Today
start_date = '2020-01-01'
interval = '1d'

# %% Getting S&P 500 index and saving it
sp500 = yf.Ticker("^GSPC")

sp500_hist = sp500.history(start=start_date, end=end_date, interval=interval)
print(sp500_hist)

# %% Volatility
# Function to calculate historical volatility
def calculate_volatility(prices, window_size=30):
    log_returns = np.log(prices[1:] / prices[:-1])
    
    volatility = np.zeros_like(prices)
    volatility[:window_size] = np.std(log_returns[:window_size])
    
    for i in range(window_size, len(prices)):
        volatility[i] = np.std(log_returns[i - window_size + 1:i + 1])
    
    return volatility

sp500_hist['Volatility'] = calculate_volatility(sp500_hist['Close'].values)

# %% Save
sp500_hist.to_csv("SP500_index.csv")

# %% Plotting price
plt.figure(figsize=(12, 6))
sns.lineplot(data=sp500_hist, x=sp500_hist.index, y='Volatility', label='Volatility', color='b')
plt.xlabel('Date')
plt.title('Novo Nordisk closing price')
plt.legend()
plt.show()

# %% Get news
from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key="b973a0633202458792a421940fc2c1be")

# /v2/top-headlines
news = newsapi.get_everything(
    q='stock market',
    from_param="2023-10-27", # Earliest available
    to=end_date,
    language='en'
    )

articles = news["articles"]
articles[1]

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
news_df.to_csv("News.csv")