## Main point:
## SoTA sentiment analysis using transformers, is readily available with just a few lines of code.
## Runs on CPU (a few minutes for the whole book) or on a free Colab GPU.

import random
import requests
import re
import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Load the sentiment analysis pipeline (GPU if there is one, otherwise CPU)
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device = device, model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

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
        sentences = [s.strip() for s in sentences if s.strip()]  # Drop empty strings
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
    # Passing the whole list lets the pipeline batch the sentences, which is much
    # faster than one call per sentence. truncation=True cuts the few very long
    # 'sentences' (whole paragraphs) to the model's maximum of 512 tokens.
    results = classifier(sentences, batch_size = 32, truncation = True)
    return [get_positive_score(x) for x in results]

if __name__ == "__main__":
    # 1. Your computer can read
    sentence = "That would have been splendid. Absolutely amazing. But it was quite the opposite."
    print(f"Sentence: \"{sentence}\"")
    print("---> Sentiment:")
    print(classifier(sentence))

    # 2. Your computer can easily read an entire book
    sentences = get_notes_from_the_underground()
    sentiment_scores = get_sentiment(sentences)

    # Calculate the moving average of the sentiment scores
    window_size = 50
    moving_average = pd.Series(sentiment_scores).rolling(window=window_size).mean()

    # Visualize the sentiment evolution
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average, label=f'Sentiment Moving Average (window size = {window_size})', color='blue')
    plt.title('Sentiment Evolution in "Notes from the Underground"')
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()

    # Save the plot
    plt.savefig('Lecture 1 - Introduction/sentiment_evolution_notes_from_the_underground.png')
    # Dostoevsky is filled with ups and downs, but ultimately leans towards the negative side...
    # Negative language is hidden in heavily contextual sentences. But the model is able to pick up on it!

    # Now you can get the entire plotline without having to read the book! (Almost at least for sentiment...)

    # 3. But is it right?
    # Print 20 random sentences with the model's label. Label them yourself first,
    # then compare. Typical misses: irony, rhetorical questions, and sentences that
    # are neutral but get forced into POSITIVE or NEGATIVE (there is no neutral class).
    random.seed(20261026)
    for i in random.sample(range(len(sentences)), 20):
        label = "POSITIVE" if sentiment_scores[i] > 0 else "NEGATIVE"
        text = " ".join(sentences[i].split())  # One line per sentence
        print(f"[{label} {sentiment_scores[i]:+.2f}] {text[:150]}")
