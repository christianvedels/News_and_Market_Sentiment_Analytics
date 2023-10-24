# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:05:45 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
from transformers import pipeline
import requests # To read .txt from URL
import re # Regular expressions

#%% Proof of concept
classifier = pipeline("sentiment-analysis"
                      , padding=True, truncation=True, max_length=512 # To make it work on long pieces of text
                      )

classifier("That would have been splendid. Absoloutly amazing. But it was quite the opposite.")

#%% Wrapping it in function
def GetSentiments(x):
    '''Returns sentiment score for x paragraph''' # Doc string
    
    # x = "That would have been great. It was not."
    out = classifier(x)

    # Handle positve / negative labels
    # Now the code returns the label (POSITIVE / NEGATIVE) and the 
    # probability of this label. Instead we want a single score where
    # 0=Completely negative, 1=Completely positive
    res = []
    
    for x in out:
        if x["label"] == "POSITIVE":
            res.append(x['score'])
        elif x["label"] == "NEGATIVE":
            res.append(1 - x['score'])
        else:
            raise Exception(x["label"]+"This should not be possible")
        
    return res

# %% Test
x = [
     "That would have been great. It was not.",
     "Heteroscedasticity is hard to pronounce",
     "This is the best I have ever encountered."
     ]
GetSentiments(x)

# %%
# Jane Austen Pride and Prejudice 
url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
response = requests.get(url)
if response.status_code == 200:
    response.encoding = 'utf-8'
    text = response.text.split('\r\n\r\n')  # Assuming paragraphs are separated by two newlines
else:
    print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")

# with open(file_path, 'r', encoding='utf-8') as file:
    # text = file.read().split('\n\n') # Assuming paragraphs are separated by two newlines

# %% Running everything
AustenSentiment = GetSentiments(text)
    
# %% Plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create a DataFrame for the sentiment scores
df = pd.DataFrame(
    {'Sentiment Score': AustenSentiment, 'Paragraph': text}   
    )

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x=df.index, y='Sentiment Score', label='Sentiment Score', color='b')
plt.xlabel('Paragraph Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis of Jane Austen Pride and Prejudice')
plt.legend()
plt.show()

# %% EXTRA: Plot moving average and text cleaning
# Text cleaning (only book)

# Finding start of ch. 1
opening = r"It is a truth universally acknowledged, that a single man in possession"
end = "her into Derbyshire, had been the means of uniting them."
found_opening = [i for i, text in enumerate(text) if re.search(opening, text)][0]
found_end = [i for i, text in enumerate(text) if re.search(end, text)][0]

text_clean = text[found_opening:found_end]
sentiment_clean = AustenSentiment[found_opening:found_end]

# MA and plot
import numpy as np
AustenMA = []
Window = 50
for i in range(len(sentiment_clean)):
    # Handle i < window
    if i < Window:
        Window_i = i
    else:
        Window_i = Window
        
    MA_i = np.mean(AustenSentiment[i-Window_i:i])
    AustenMA.append(MA_i)
    
# Create a DataFrame for the sentiment scores AND MA
df = pd.DataFrame(
    {
     'Sentiment Score': sentiment_clean, 
     'Paragraph': text_clean,
     'Moving average sentiment': AustenMA
     }   
    )

# Create a scatter plot with a smoothed line
titletxt = f"Sentiment Analysis of Jane Austen Pride and Prejudice\nMoving average with window of {Window} paragraphs"
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Moving average sentiment', label='Moving average sentiment', color='b')
plt.xlabel('Paragraph Index')
plt.ylabel('Sentiment Score (MA)')
plt.title(titletxt)
plt.legend()
plt.show()
    


    

            
        
        
        
    


