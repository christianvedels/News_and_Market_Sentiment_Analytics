# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:00:55 2024

@author: christian-vs
"""


# Please note that ChatGPT can solve this exercise, which limits your job
# to just understanding the code you are excecuting: https://chatgpt.com/share/672b4670-31e4-800c-958f-40311d19263e

import requests
import re
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Load the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", device = "cuda")

def get_notes_from_the_underground():
    url = 'https://www.gutenberg.org/cache/epub/600/pg600.txt'
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        text = response.text

        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\r\n\r\n', text)
        # Clean text
        sentences = sentences[11:]  # Remove the first few lines
        sentences = sentences[:-117]  # Remove the last few lines
        return sentences
    else:
        raise Exception(f"Failed to retrieve data from {url}. Status code: {response.status_code}")

def get_positive_score(x):
    if x["label"] == "POSITIVE":
        res_x = x['score']
    elif x["label"] == "NEGATIVE":
        res_x = 1 - x['score']
    else:
        raise Exception(x["label"] + " This should not be possible")

    res_x = res_x * 2 - 1  # Scale to -1 to 1
    return res_x

def get_sentiment(sentences):
    sentiment_scores = [get_positive_score(classifier(sentence)[0]) for sentence in sentences]
    return sentiment_scores

if __name__ == "__main__":
    print(classifier("That would have been splendid. Absolutely amazing. But it was quite the opposite."))
    
    # Retrieve the sentences from "Notes from the Underground"
    sentences = get_notes_from_the_underground()
    
    # Get the sentiment scores for the sentences
    sentiment_scores = get_sentiment(sentences)
    
    # Calculate the moving average of the sentiment scores
    window_size = 50
    moving_average = pd.Series(sentiment_scores).rolling(window=window_size).mean()
    
    # # Visualize the sentiment evolution
    # plt.figure(figsize=(12, 6))
    # plt.plot(moving_average, label='Sentiment Moving Average (window size = 50)', color='blue')
    # plt.title('Sentiment Evolution in "Notes from the Underground"')
    # plt.xlabel('Sentence Index')
    # plt.ylabel('Sentiment Score (-1 to 1)')
    # plt.axhline(0, color='gray', linestyle='--')
    # plt.legend()
    # plt.show()
