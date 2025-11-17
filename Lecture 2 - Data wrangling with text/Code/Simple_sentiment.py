import pandas as pd
import spacy
import matplotlib.pyplot as plt

# import os
# os.chdir(r"D:\Dropbox\Teaching\News_and_Market_Sentiment_Analytics\2025\News_and_Market_Sentiment_Analytics")

# Define a function to calculate sentiment score using AFINN
def afinn_sentiment(text):
    doc = nlp(text)
    sentiment_score = 0
    for token in doc:
        word = token.text.lower()
        if word in afinn_dict:
            sentiment_score += afinn_dict[word]
    return sentiment_score

if __name__ == "__main__":
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load the AFINN lexicon
    afinn = pd.read_csv("https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt", sep="\t", header=None, names=["word", "score"])
    afinn_dict = dict(zip(afinn["word"], afinn["score"]))
    
    # Load the news data
    data = pd.read_csv('Lecture 2 - Data wrangling with text/Code/Novo_Nordisk_News.csv')

    # Apply the sentiment function to each headline
    data['sentiment_score'] = data['Headline'].apply(afinn_sentiment)
    
    # Ensure date is in datetime format, coerce errors to NaT
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    data = data.dropna(subset=['Date'])
    
    # Filter out rows with the default date of '1970-01-01'
    data = data[data['Date'] != '1970-01-01']

    # Ensure date is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort the data by date
    data = data.sort_values(by='Date')
    
    # Plot sentiment scores over time
    plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['sentiment_score'], color='blue', label='Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Score Over Time (AFINN)')
    plt.legend()
    plt.savefig("Lecture 2 - Data wrangling with text/Code/sentiment_over_time.png")
