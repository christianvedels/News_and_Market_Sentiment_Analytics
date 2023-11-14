# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:56:51 2023

@author: chris
"""
# %% Dir
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
import pandas as pd
import numpy as np
import random as r
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data
male   = pd.read_table("../Data/names_males.txt", sep = "\t", encoding='ISO-8859-1')
female = pd.read_table("../Data/names_females.txt", sep = "\t", encoding='ISO-8859-1') 

# %% Data wrangling
# Drop NA 
male = male.dropna()
female = female.dropna()

# Upsample
male = pd.DataFrame({
    'Navn': male['Navn'].repeat(male['ANTAL'])
})

female = pd.DataFrame({
    'Navn': female['Navn'].repeat(female['ANTAL'])
})

# Names to lower
male['Navn'] = [i.lower() for i in male['Navn'].tolist()]
female['Navn'] = [i.lower() for i in female['Navn'].tolist()]

# Add gender labels to the data
male['gender'] = 'male'
female['gender'] = 'female'

# Combine male and female datasets
list_males   = list(zip(male['Navn'], male['gender']))
list_females = list(zip(female['Navn'], female['gender']))
labeled_names = list_males + list_females

# Shuffle
r.seed(20)
r.shuffle(labeled_names)

# Features
X = [name for (name, gender) in labeled_names]
y = [gender for (name, gender) in labeled_names]

# # Convert labels
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
label_encoder_X = LabelEncoder()
X = label_encoder_X.fit_transform(X)

# # Reshape X
X = np.array(X).reshape(-1, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# %% Classifier
# Initialize and train the scikit-learn Naive Bayes classifier
classifier = CategoricalNB()
classifier.fit(X_train, y_train)

# %%
# Making classifier wrapper function
def classifier_wrapper(x):
    # Contain all pre/post processing steps*
    x = [i.lower() for i in x]
    x = label_encoder_X.transform(x)
    x = np.array(x).reshape(-1, 1)
    res = classifier.predict(x)
    res = label_encoder_y.inverse_transform(res)
    return res

# %%
# Classify some examples

# Errors because of unseen:
print(classifier_wrapper(['Neo']))
print(classifier_wrapper(['Trinity']))
print(classifier_wrapper(['Casper']))
print(classifier_wrapper(['Marie']))

 # Name of Elon Musk' child
# print(classifier_wrapper(["XÆA-12"]))
# Error because it is unseen in traning

# %% Evaluate
# Evaluate the accuracy on the test set
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix
labels = ['female', 'male']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Normalize the confusion matrix to percentages
conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
print('Normalized Confusion Matrix:')
print(conf_matrix_pct)

# Plot Normalized Confusion Matrix
labels = ['female', 'male']
sns.heatmap(conf_matrix_pct, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix (%)')
plt.show()

# %% Test on Danish census data from the year 1787 from Link Lives
# https://link-lives.dk/en/about-link-lives/
# This used to not contain gender. Now it does.
df1787 = pd.read_csv("../Data/Names_gender1787.csv")
print(df1787)

# Take only first of first names
df1787['clean_names'] = df1787['first_names'].str.split().str[0]

# Remove unknown names
df1787_clean = df1787[df1787['clean_names'].isin(male['Navn']) | df1787['clean_names'].isin(female['Navn'])]
df1787_removed = df1787[~(df1787['clean_names'].isin(male['Navn']) | df1787['clean_names'].isin(female['Navn']))]

# Print removed/unremoved
def N_of(x): 
    return np.sum(x['n']) 
N_all = N_of(df1787)
N_removed = N_of(df1787_removed) 
N_kept = N_of(df1787_clean)

# Print descriptive
print(f"All obs: {N_all}")
print(f"Obs. kept: {N_kept}; Pct: {100*round(N_kept/N_all, 4)}%")
print(f"Obs. removed: {N_removed}; Pct: {100*round(N_removed/N_all, 4)}%")

# Transform to features
X = label_encoder_X.transform(df1787_clean['clean_names'])
# Reshape X
X = np.array(X).reshape(-1, 1)

# Predict 
y_pred = classifier.predict(X)

# ==== Test performance ====
# True labels 1787
y = label_encoder_y.transform(df1787_clean['gender'])

# Accuracy estimated with weights from population counts in 1787
weights = df1787_clean['n'].tolist()
accuracy = accuracy_score(y, y_pred, sample_weight = weights)
print(f"\nAccuracy of Naïve Bayes model on 1787 data: {100*round(accuracy, 4)}%")
print(f"Corrected for unfound names: {100*round(accuracy*N_kept/N_all, 4)}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred, sample_weight = weights)
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix
labels = ['female', 'male']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Normalize the confusion matrix to percentages
row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
conf_matrix_pct = conf_matrix.astype('float') / row_sums
print('Normalized Confusion Matrix:')
print(conf_matrix_pct)

# Plot Normalized Confusion Matrix
labels = ['female', 'male']
sns.heatmap(conf_matrix_pct, annot=True, fmt='.4f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix (%)')
plt.show()

