# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:29:54 2023

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

# %% Corpus
words_doc = nltk.Text(gutenberg.words(gutenberg.fileids()))
# Run the following instead if the entire corpus is too large for your computer
# words_doc = nltk.Text(nltk.corpus.gutenberg.words('carroll-alice.txt'))

# %% Stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %% Lowercase, remove stopwords, lemmatize
words_doc = [word.lower() for word in words_doc if word.isalpha()]

# We could also consider running it without this
words_doc = [word for word in words_doc if word not in stop_words]
words_doc = [lemmatizer.lemmatize(word) for word in words_doc]

# %% Calculate the frequency of the words inside
freqs = FreqDist(words_doc)

# Convert to list
freq_dist_list = [(word, freq) for word, freq in freqs.items()]

df = pd.DataFrame(freq_dist_list, columns=['Word', 'Frequency'])

# Sort by frequency
df = df.sort_values(by='Frequency', ascending=False)

# Reset the index to have consecutive integers as the index
df = df.reset_index(drop=True)

df['log_Frequency'] = np.log(df['Frequency'])
df['log_Index'] = np.log(df.index + 1)

# %% Make plot
small = df[0:200]

plt.figure(figsize = (12, 6))
sns.lineplot(data = small, x = 'log_Index', y = "log_Frequency")
plt.xlabel('log(Rank)')
plt.title('log(Frequency)')
plt.show()

# The most common words are overrepresented (turns out to be a common issue)

# %% Regression
# Reshape the data into a 2D array
X = df['log_Index'].values.reshape(-1, 1)
y = df["log_Frequency"]

# Add a constant (intercept) to the X variable
X = sm.add_constant(X)

# Calculate the weights as the inverse of rank (more information)
weights = 1.0 / (df.index+1)

# Create and fit the WLS (Weighted Least Squares) regression
model = sm.WLS(y, X, weights).fit()

# Print the regression summary table
print(model.summary())

model.rsquared

# %% Plot of residuals - something funky is happening here
# But maybe we can just ignore this for now and hope that the statitics gods
# will not punish us.

# Calculate residuals
residuals = model.resid

# Create a scatter plot of X versus residuals
plt.scatter(X[:, 1], residuals)  # X[:, 1] contains the original X values (excluding the constant term)
plt.xlabel("log(Rank)")
plt.ylabel("Residuals")
plt.title("log(Rank) versus Residuals")
plt.show()

# %% Plot
# Create a scatter plot
sns.scatterplot(x='log_Index', y='log_Frequency', data=df, label='Data Points')

# Calculate the regression line using the estimated coefficients
X_values = df['log_Index']
y_pred = model.params['const'] + model.params['x1'] * X_values
data0 = pd.DataFrame({"X_values": X_values, "y_pred": y_pred})

# Plot the regression line
sns.lineplot(data = data0, x = "X_values", y = "y_pred", label='Regression Line')

# Add labels and a title
plt.xlabel('X (log_Index)')
plt.ylabel('Y (log_Frequency)')
plt.title('Seaborn Regression Plot')

# Show the plot
plt.legend()
plt.show()



