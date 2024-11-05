# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:08:11 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from afinn import Afinn

# Other stuff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import string

# %% Load data
news = pd.read_csv("Novo_Nordisk_News.csv")

# %% Tokenizer
headlines = news['Headline'].apply(word_tokenize)
content = news['Content'].apply(word_tokenize)

# %% To lower
headlines = [[word.lower() for word in x] for x in headlines]
content = [[word.lower() for word in x] for x in content]

# %% Remove stopwords
stopwords = set(stopwords.words('english'))
headlines = [[word for word in x if word not in stopwords] for x in headlines]
content = [[word for word in x if word not in stopwords] for x in content]

# %% Remove punctuation
headlines = [[word for word in x if word not in string.punctuation] for x in headlines]
content = [[word for word in x if word not in string.punctuation] for x in content]

# %% Load sentiments from AFINN
# Load AFINN sentiment lexicon
afinn = Afinn()

# %% Function for sentiment
def Sentiment(x, verbose=False):
    sentiment = 0
    for word in x:
        # Find word in AFINN data
        afinn_score = afinn.score(word)
        
        # Add it to sentiment
        sentiment += afinn_score
        
        if verbose:
            print(f'Overall sentiment: {sentiment}, word: {word}, score {afinn_score}')
    
    # Normalize by sequence length
    sentiment = sentiment / len(x)
    
    return sentiment


# %% Test function
x = headlines[0]
Sentiment(x, True)

# %% Run on all
headlines_sent = [Sentiment(x) for x in headlines]
content_sent = [Sentiment(x) for x in content]

# %% Add to df
news['Headline_sentiment'] = headlines_sent
news['Content_sentiment'] = content_sent

# Look at results!
# %% Correlation between headline and content:
plt.figure(figsize=(12, 6))
sns.scatterplot(data=news, x='Headline_sentiment', y='Content_sentiment', color='b')
plt.title('Novo Nordisk sentiment')
plt.legend()
plt.show()    

# %% Save results
news.to_csv("Novo_Sentiment_AFINN.csv")
        
        
    
    