# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:14:08 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import spacy

from transformers import pipeline

# %% Read data
news = pd.read_csv("News.csv")
sp500 = pd.read_csv("SP500_index.csv")

# %% Language model
nlp = spacy.load("en_core_web_sm")

# %% Get_data_date
# Recursive function

def Get_data_date(d, recursions = 0):
    d_date = d.date()
    res_d = news[news['Date0'] == d_date]
    res_d['Recursions'] = recursions

    # If no data call on previous date
    if res_d.shape[0] == 0:
        previous_date = d - pd.Timedelta(days=1)
        recursions =+ 1
        res_d = Get_data_date(previous_date, recursions)
        
    return res_d

# %% Features
def Get_features(text):
    doc = nlp(text)
    
    verbs          = [token.text.lower() for token in doc if token.pos_ == 'VERB']
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    organiations   = [org[0].lower() for org in named_entities if org[1]=="ORG"]
    geography      = [gpe[0].lower() for gpe in named_entities if gpe[1]=="GPE"]
    
    # Append other results
    res = verbs + organiations + geography
    
    # Return flat unique list
    return list(set(res))

# %% Get sentiments
classifier = pipeline("sentiment-analysis")

def Get_sentiments(x):
    '''Returns sentiment score for x paragraph''' # Doc string
    
    # Truncate or pad the input to match the model's expected size
    max_length = 512  # Adjust this based on the model's maximum input size
    x = x[:max_length]
    
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

# %% Extracting features 
news['Date'] = pd.to_datetime(news['Date'])
news['Date0'] = news['Date'].dt.date
earliest_date = min(news['Date0'])

# Make empty data frame from 'earliest_date' until today
today = datetime.today().date()
date_range = pd.date_range(earliest_date, today, freq='D')

# Feature extraction loop
word_features = []
targets = [] # Volatility
sentiments = []

for d in date_range:
    data_d = Get_data_date(d)
    
    text_d = ' '.join(data_d['Content'])
    word_features.append(Get_features(text_d))
    sentiments.append(Get_sentiments(text_d)[0])
    
    # Extract targets 

    
# %% Train model (Random Forest)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

# Combine word features into a single string for CountVectorizer
word_features_str = [' '.join(features) for features in word_features]

# Vectorize the text features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(word_features_str)

# Convert sentiments list to NumPy array
y_targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_targets, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %% Visualize Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Random Forest Model: Actual vs. Predicted Sentiments')
plt.xlabel('Actual Sentiments')
plt.ylabel('Predicted Sentiments')
plt.show()




