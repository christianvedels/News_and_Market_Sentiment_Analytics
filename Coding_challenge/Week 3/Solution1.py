# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:57:36 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
from transformers import pipeline
import random as r

#%% Setup
classifier = pipeline("zero-shot-classification"
                      , padding=True, truncation=True, max_length=512, # To make it work on long pieces of text
                       # model = "typeform/distilbert-base-uncased-mnli"
                      )
candidate_labels = ["Joy", "Anger", "Surprise", "Sadness", "Fear", "Confidence"]

classifier(
    "And then something dreadful happenened. I wouldn't dare to rethink it if I had no need.",
    candidate_labels
    )

#%% Wrapping it in function
def GetEmotions(x):
    candidate_labels = ["Happy", "Anxious", "Excited", "Frustrated", "Sad", "Hopeful"]
    res = classifier(x, candidate_labels)
    
    sorted_results_dict = {}
    for i, text in enumerate(x):
        label_score_mapping = {label: score for label, score in zip(res[i]['labels'], res[i]['scores'])}

        # Sort the labels and scores by candidate_labels
        sorted_labels = [label for label in candidate_labels]
        sorted_scores = [label_score_mapping.get(label, 0.0) for label in candidate_labels]
        sorted_results_dict[text] = {"labels": sorted_labels, "scores": sorted_scores}
    
    return sorted_results_dict

# %% Test
x = [
     "That would have been great. It was not.",
     "Heteroscedasticity is hard to pronounce",
     "This is the best I have ever encountered."
     ]
GetEmotions(x)

# %% Load data
# Load the NLTK Twitter dataset
import nltk
from nltk.corpus import twitter_samples

nltk.download("twitter_samples")
tweets_neg = twitter_samples.strings("negative_tweets.json")
tweets_pos = twitter_samples.strings("positive_tweets.json")

# Randomly select a subset of tweets
r.seed(20)
r.shuffle(tweets_neg)
r.shuffle(tweets_pos)
tweets_neg = tweets_neg[:20]
tweets_pos = tweets_pos[:20]

# %% Running everything
emotions_neg = GetEmotions(tweets_neg)
emotions_pos = GetEmotions(tweets_pos)

# %% Plot it
# Import necessary libraries for the final steps
import matplotlib.pyplot as plt
import numpy as np

# Calculate the average emotions for negative tweets
average_emotions_neg = np.mean(np.array([data['scores'] for _, data in emotions_neg.items()]), axis=0)

# Calculate the average emotions for positive tweets
average_emotions_pos = np.mean(np.array([data['scores'] for _, data in emotions_pos.items()]), axis=0)

# Subtract mean
mean_emotions = (average_emotions_neg + average_emotions_pos)/2
average_emotions_neg = average_emotions_neg - mean_emotions
average_emotions_pos = average_emotions_pos - mean_emotions

# Labels for emotions
emotions_labels = candidate_labels

# Define consistent colors for emotions
emotion_colors = ['green', 'red', 'blue', 'purple', 'gray', 'orange']

# Create separate bar plots for negative and positive emotions with consistent colors
plt.figure(figsize=(12, 5))

# Negative tweets plot
plt.subplot(1, 2, 1)
plt.bar(emotions_labels, average_emotions_neg, color=emotion_colors)
plt.xlabel("Emotion")
plt.ylabel("Average Score")
plt.title("Average Emotions in Negative Tweets")

# Positive tweets plot
plt.subplot(1, 2, 2)
plt.bar(emotions_labels, average_emotions_pos, color=emotion_colors)
plt.xlabel("Emotion")
plt.ylabel("Average Score")
plt.title("Average Emotions in Positive Tweets")

plt.tight_layout()
plt.show()



    

            
        
        
        
    


