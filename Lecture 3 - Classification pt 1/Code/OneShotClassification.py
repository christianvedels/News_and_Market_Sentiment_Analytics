# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:49:03 2023

@author: chris
"""

from transformers import pipeline

# Initialize the zero-shot classification pipeline with GPT-3
classifier = pipeline("zero-shot-classification")

# Example news headlines or tweets
texts = [
    "Investors celebrate as the stock market reaches an all-time high.",
    "Economic uncertainty leads to anxiety among traders.",
    "Tech company's breakthrough innovation sparks excitement in the market.",
    "Global trade tensions cause frustration for investors.",
    "News of company layoffs generates sadness among employees.",
    "Promising healthcare developments bring hope for the future.",
]

# Candidate emotion labels
candidate_labels = ["Happy", "Anxious", "Excited", "Frustrated", "Sad", "Hopeful"]

# Perform zero-shot classification for emotion analysis
results = classifier(texts, candidate_labels)

# Print the results
for i, text in enumerate(texts):
    print(f"News Headline/Tweet: {text}")
    for j, label in enumerate(candidate_labels):
        print(f"Emotion: {label}, Score: {results[i]['scores'][j]:.3f}")
    print()
