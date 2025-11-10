# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:56:51 2023

@author: chris
"""

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
import unidecode

# %% Functions
def clean_strings(x):
    # Cast to string
    x = [str(y) for y in x] # Cast to string
    x = [y.lower() for y in x] # Lower case
    x = [unidecode.unidecode(y) for y in x] # Remove weird letters
    
    return(x)

def clean_data(male, female):
    # Drop NA 
    male = male.dropna()
    female = female.dropna()
    
    # Upsample
    male = pd.DataFrame({
        'clean_name': male['clean_name'].repeat(male['ANTAL'])
    })
    
    female = pd.DataFrame({
        'clean_name': female['clean_name'].repeat(female['ANTAL'])
    })
        
    # Add gender labels to the data
    male['gender'] = 'male'
    female['gender'] = 'female'
    
    # Combine male and female datasets
    list_males   = list(zip(male['clean_name'], male['gender']))
    list_females = list(zip(female['clean_name'], female['gender']))
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
    
    # Return this and the encoders
    return X_train, X_test, y_train, y_test, label_encoder_y, label_encoder_X

# Making classifier wrapper function (will be used to apply the classificer easily)
def classifier_wrapper(x):
    # NOTE: label_encoder_y and label_encoder_X must be available in environment
    # Contain all pre/post processing steps*
    x = [i.lower() for i in x]
    x = label_encoder_X.transform(x)
    x = np.array(x).reshape(-1, 1)
    res = classifier.predict(x)
    res = label_encoder_y.inverse_transform(res)
    return res

# %% Main
if __name__ == "__main__":
    
    # Load modern names data
    male   = pd.read_table("Data/names_males.txt", sep = "\t", encoding='ISO-8859-1')
    female = pd.read_table("Data/names_females.txt", sep = "\t", encoding='ISO-8859-1') 
    
    male['clean_name'] = clean_strings(male.Navn.tolist())
    female['clean_name'] = clean_strings(female.Navn.tolist())
    
    # Get train / test data and encoders
    X_train, X_test, y_train, y_test, label_encoder_y, label_encoder_X = clean_data(male, female)
    
    # Initialize and 'train' the scikit-learn Naive Bayes classifier
    classifier = CategoricalNB()
    classifier.fit(X_train, y_train) # Calculates probabilities
    
    # Classify some examples
    print(classifier_wrapper(['Neo']))
    print(classifier_wrapper(['Trinity']))
    print(classifier_wrapper(['Casper']))
    print(classifier_wrapper(['Marie']))
    
    # Name of Elon Musk' child
    try: 
        print(classifier_wrapper(["XÆA-12"]))
    except:
        print("Name not in training data")
        
    # Evaluate the accuracy on the test set
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))
        
    # Apply to Danish census data from the year 1787 from Link Lives
    # https://link-lives.dk/en/about-link-lives/
    # This used to not contain gender. But we will use it as a test case anyway
    df1787 = pd.read_csv("Data/Names_gender1787.csv")
    print(df1787)
    
    # Take only first of first names
    df1787['clean_name'] = df1787['first_names'].str.split().str[0]
    df1787['clean_name'] = clean_strings(df1787.clean_name)
    
    # Remove unknown names
    df1787_clean = df1787[df1787['clean_name'].isin(male['clean_name']) | df1787['clean_name'].isin(female['clean_name'])]
    df1787_removed = df1787[~df1787['clean_name'].isin(male['clean_name']) & ~df1787['clean_name'].isin(female['clean_name'])]

    # Print removed/unremoved
    def N_of(x): 
        return np.sum(x['n']) 
    N_all = N_of(df1787)
    N_kept = N_of(df1787_clean)

    # Print descriptive
    print(f"All obs: {N_all}")
    print(f"Obs. kept: {N_kept}; Pct: {100*round(N_kept/N_all, 4)}%")
    print(f"Obs. removed: {N_all - N_kept}; Pct: {100*round((N_all - N_kept)/N_all, 4)}%")

    # Transform to features
    X = label_encoder_X.transform(df1787_clean['clean_name'])
    # Reshape X
    X = np.array(X).reshape(-1, 1)

    # Predict 
    y_pred = classifier.predict(X)

    # True labels 1787
    y = label_encoder_y.transform(df1787_clean['gender'])

    # Accuracy estimated with weights from population counts in 1787
    weights = df1787_clean['n'].tolist()
    accuracy = accuracy_score(y, y_pred, sample_weight = weights)
    
    print(f"\nAccuracy of Naïve Bayes model on 1787 data: {100*round(accuracy, 4)}%")
    print(f"Corrected for unfound names: {100*round(accuracy*N_kept/N_all, 4)}%")

