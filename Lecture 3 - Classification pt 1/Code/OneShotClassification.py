# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:49:03 2023

@author: chris
"""

from transformers import pipeline

# Initialize the zero-shot classification pipeline
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

# BUGFIX
# Sort according to 'candidate_labels'
# Create a mapping from the labels to their scores
sorted_results_dict = {}
for i, text in enumerate(texts):
    label_score_mapping = {label: score for label, score in zip(results[i]['labels'], results[i]['scores'])}

    # Sort the labels and scores by candidate_labels
    sorted_labels = [label for label in candidate_labels]
    sorted_scores = [label_score_mapping.get(label, 0.0) for label in candidate_labels]
    sorted_results_dict[text] = {"labels": sorted_labels, "scores": sorted_scores}

# Print the results
for text, data in sorted_results_dict.items():
    print(f"News Headline/Tweet: {text}")
    for label, score in zip(data["labels"], data["scores"]):
        print(f"Emotion: {label}, Score: {score:.3f}")
    print()
