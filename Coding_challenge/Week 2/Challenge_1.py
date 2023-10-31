# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 20:14:19 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
import nltk

from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import statsmodels.api as sm

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

#%%
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(x):
    x = [word.lower() for word in x if word.isalpha()]
    x = [word for word in x if word not in stop_words]
    x = [lemmatizer.lemmatize(word) for word in x]
    return x

# %% Load text
def read_text_file(FileID = 0): 
    file_path = f'Texts/Text{FileID}.txt'
    with open(file_path, 'r') as f: 
        text = f.read().split(' ')
    return text

# Test
text0 = read_text_file(0)
clean_text(text0)

# %% Freuquencies
def frequencies(x):
    freqs = FreqDist(x)

    # Convert to list
    freq_dist_list = [(word, freq) for word, freq in freqs.items()]

    df = pd.DataFrame(freq_dist_list, columns=['Word', 'Frequency'])

    # Sort by frequency
    df = df.sort_values(by='Frequency', ascending=False)

    # Reset the index to have consecutive integers as the index
    df = df.reset_index(drop=True)

    df['log_Frequency'] = np.log(df['Frequency'])
    df['log_Index'] = np.log(df.index + 1)
    
    return df

# Test
text0 = read_text_file(0)
text0 = clean_text(text0)
frequencies(text0)

# %% Make plot
def make_plot(FileID = 0):
    text0 = read_text_file(FileID)
    text0 = clean_text(text0)
    df = frequencies(text0)
    
    small = df[0:200]

    plt.figure(figsize = (12, 6))
    sns.lineplot(data = small, x = 'log_Index', y = "log_Frequency")
    plt.xlabel('log(Rank)')
    plt.title('log(Frequency)')
    plt.show()

make_plot(0)

# %% Regression
def ZipfIt(FileID = 0):
    
    # ==> Preprocess data
    # Read files
    text0 = read_text_file(FileID)
    # Clean text
    text0 = clean_text(text0)
    # Calculate frequencies
    df = frequencies(text0)
    
    # ==> Run regressions
    # Reshape the data into a 2D array
    X = df['log_Index'].values.reshape(-1, 1)
    y = df["log_Frequency"]

    weights = 1.0 / (df.index+1)
    # Add a constant (intercept) to the X variable
    X = sm.add_constant(X)

    # Create and fit the WLS (Weighted Least Squares) regression
    model = sm.WLS(y, X, weights).fit()
    
    slope = model.params[1]
    
    return slope

ZipfIt(0)

# %% Run estimate for all files
answers = pd.read_csv("Answer.csv")
# Run it
answers["Slope"] = [ZipfIt(i) for i in answers["TextID"]]

# %% Histogram of slopes
plt.figure(figsize = (12, 8))
plt.hist(answers["Slope"], 30)
plt.show() # Has two distinct humps

# %% Classifier class
class Classifier:
    def __init__(self, slopes, sd = 2):
        self.std = np.std(slopes)
        self.median = np.median(slopes)
        self.sd = sd
        print(f'CI: {self.median-sd*self.std}, {self.median+sd*self.std}')
        
    def classify(self, slope):
        std = self.std
        
        # According to Zipf law:
        theoretical_slope = self.median
        
        # If more than 2 std away from theoretical value
        # Then it is probably not a natural text
        cutoff = theoretical_slope + self.sd*std
        prediction = slope < cutoff
        return(prediction)

# Test        
classifier = Classifier(answers["Slope"])
classifier.classify(-1)

# %% Run
classifier = Classifier(answers["Slope"], sd = 1)
answers["Prediction"] = [classifier.classify(i) for i in answers["Slope"]]

# Test
np.mean(answers["Real"] == answers["Prediction"])

# %% Histogram of slopes
plt.figure(figsize = (12, 8))
plt.hist(answers["Slope"], 30)
plt.show() # Has two distinct humps     