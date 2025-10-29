## Main point:
## SoTA sentiment analysis using transformers, is readily available with just a few lines of code.

import requests
import re
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Load the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", device = "cuda", model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

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
    sentiment_scores = [get_positive_score(classifier(sentence)[0]) for sentence in sentences] # This will trigger a warning [1]
    # [1] Warning is about sequential model calls not being efficient. For simplicity, we ignore it here.
    return sentiment_scores

if __name__ == "__main__":
    print("Sample sentiment analysis:")
    sentence = "That would have been splendid. Absolutely amazing. But it was quite the opposite."
    print(f"Sentence: \"{sentence}\"")
    print("---> Sentiment:")
    print(classifier(sentence))

    input("Press Enter to continue to the full analysis of \"Notes from the Underground\"...")

    # Retrieve the sentences from "Notes from the Underground"
    sentences = get_notes_from_the_underground()
    
    # Get the sentiment scores for the sentences
    sentiment_scores = get_sentiment(sentences)
    
    # Calculate the moving average of the sentiment scores
    window_size = 50
    moving_average = pd.Series(sentiment_scores).rolling(window=window_size).mean()
    
    # Visualize the sentiment evolution
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average, label='Sentiment Moving Average (window size = 50)', color='blue')
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
