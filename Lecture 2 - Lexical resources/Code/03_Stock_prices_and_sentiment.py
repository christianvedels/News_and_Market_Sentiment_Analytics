# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:32:49 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
# Other stuff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# %% Load data
novo = pd.read_csv('Novo_Nordisk_prices.csv')
sp500 = pd.read_csv('SP500_index.csv')
sentiment = pd.read_csv('Novo_Sentiment_AFINN.csv')

# %% Check sources
set(sentiment['Source'])
# Seems fine

# %% Format as date
sentiment['Date'] = pd.to_datetime(sentiment['Date'])
novo['Date'] = pd.to_datetime(novo['Datetime'])
sp500['Date'] = pd.to_datetime(sp500['Datetime'])

# %% Filter out 'removed'
sentiment = sentiment[sentiment['Content'] != '[Removed]']

# %% Plot sentiment over time

# The following is a mess but it makes sense, when you look at data for just one day
# filter_date = pd.to_datetime('2023-10-18')
# sentiment = sentiment[sentiment['Date']>filter_date]

plt.figure(figsize=(12, 6))
sns.lineplot(data=sentiment, x='Date', y='Headline_sentiment', label='Headline', color='b')
sns.lineplot(data=sentiment, x='Date', y='Content_sentiment', label='Content', color='r')
plt.xlabel('Date')
plt.title('Novo Nordisk sentiment')
plt.legend()
plt.show()

# %% Price changes
novo['Rel_change'] = np.log(novo['Close']) - np.log(novo['Open'])
sp500['Rel_change'] = np.log(sp500['Close']) - np.log(sp500['Open'])

# The following is a mess but it makes sense, when you look at data for just one day
# filter_date = pd.Timestamp('2023-10-19', tz=timezone('US/Eastern'))
# novo = novo[novo['Date']>filter_date]
# sp500 = sp500[sp500['Date']>filter_date]

plt.figure(figsize=(12, 6))
sns.lineplot(data=novo, x='Date', y='Rel_change', label='Novo rel. Change', color='b')
sns.lineplot(data=sp500, x='Date', y='Rel_change', label='S&P500 rel. Change', color='r')
plt.xlabel('Date')
plt.title('Stock prices change')
plt.legend()
plt.show()

# Notice that they follow each other
# We are more interested in instances where novo changes are different from sp500

# %% Merge stock prices
sp500_0 = sp500[['Date', 'Rel_change']] # Limited sp500 data
# Renaming
sp500_0 = sp500_0.rename(columns={"Date": "Date", "Rel_change": "Rel_change_SP500"})

# Merging
stocks = pd.merge(novo, sp500_0, on="Date")

# Calculating relative changes to SP changes 
stocks['Rel_change_to_SP500'] = stocks['Rel_change'] - stocks['Rel_change_SP500']

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=stocks, x='Date', y='Rel_change_SP500', label='Novo rel. Change', color='b')
plt.xlabel('Date')
plt.title('Novo stock price change relative to S&P500 changes')
plt.legend()
plt.show()

# %% Add sentiment
plt.figure(figsize=(12, 6))
sns.lineplot(data=stocks, x='Date', y='Rel_change_to_SP500', label='Novo rel. change to SP500', color='b')
sns.lineplot(data=sentiment, x='Date', y='Headline_sentiment', label='Headline', color='g')
sns.lineplot(data=sentiment, x='Date', y='Content_sentiment', label='Content', color='r')
plt.xlabel('Date')
plt.title('Novo stock price change relative to S&P500 changes')
plt.legend()
plt.show()

# %% We need to normalize to get anything useful
# Standardization
def Standardization(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

sentiment["Headline_sentiment_std"] = Standardization(sentiment["Headline_sentiment"])
sentiment["Content_sentiment_std"] = Standardization(sentiment["Content_sentiment"])
stocks['Rel_change_to_SP500_std'] = Standardization(stocks['Rel_change_to_SP500'])

# %% Plot again
plt.figure(figsize=(12, 6))
sns.lineplot(data=stocks, x='Date', y='Rel_change_to_SP500_std', label='Novo rel. change to SP500', color='b')
sns.lineplot(data=sentiment, x='Date', y='Headline_sentiment_std', label='Headline', color='g')
sns.lineplot(data=sentiment, x='Date', y='Content_sentiment_std', label='Content', color='r')
plt.xlabel('Date')
plt.title('Novo stock price change relative to S&P500 changes')
plt.legend()
plt.show()

# %% Plotting in sepperate panes
# Create a figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# Plot the data in the subplots
sns.lineplot(data=stocks, x='Date', y='Rel_change_to_SP500_std', label='Novo rel. change to SP500', color='b', ax=axes[0])
axes[0].set_title('Novo stock price change relative to S&P500 changes')
axes[0].legend()

sns.lineplot(data=sentiment, x='Date', y='Headline_sentiment_std', label='Headline', color='g', ax=axes[1])
axes[1].set_title('Headline Sentiment (AFINN lexical approach)')
axes[1].legend()

sns.lineplot(data=sentiment, x='Date', y='Content_sentiment_std', label='Content', color='r', ax=axes[2])
axes[2].set_title('Content Sentiment (AFINN lexical approach)')
axes[2].legend()

# Set common x-axis label
axes[-1].set_xlabel('Date')

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('Novo_sentiment.png')

# Show the plot
plt.show()

