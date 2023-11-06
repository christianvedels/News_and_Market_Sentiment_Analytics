# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 00:30:46 2023

@author: chris
"""

# %% Libraries
import spacy
from collections import defaultdict
import nltk
from nltk.corpus import twitter_samples
import random as r

# %% Parameters
tweet_sentiment = "positive"
tweet_sentiment = "negative"

# %% Load
# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load the NLTK Twitter dataset
nltk.download("twitter_samples")
tweets = twitter_samples.strings(f"{tweet_sentiment}_tweets.json")

# Randomly select a subset of tweets
r.seed(20)
r.shuffle(tweets)
tweets = tweets[:100]

# %% Define a function to summarize tweets
def summarize_tweet(text, summary_length=2):
    
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables to store content words and their frequencies
    content_words = defaultdict(int)

    # Define a set of allowed PoS tags for content words (nouns, adjectives)
    allowed_pos_tags = {"NOUN", "ADJ"}

    # Extract content words and their frequencies
    for token in doc:
        if token.pos_ in allowed_pos_tags:
            content_words[token.text] += 1

    # Sort content words by frequency in descending order
    sorted_content_words = sorted(content_words.items(), key=lambda x: x[1], reverse=True)

    # Create a summary by selecting the top content words
    selected_words = [word for word, _ in sorted_content_words[:summary_length]]
    summary = " ".join(selected_words)

    return summary

# %% Process and summarize the tweets
tweet_summaries = [summarize_tweet(tweet, summary_length=2) for tweet in tweets]

# %%
res = ', '.join(tweet_summaries)
print(res)

