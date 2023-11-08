# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:23:35 2023

@author: christian-vs
"""

import spacy
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Define a function to scrape and summarize a news article
def summarize_news_article(url, summary_length=10):
    breakpoint()
    # Make a request to the URL and parse the HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the article text by excluding unwanted elements (e.g., video ads)
    article = soup.find('article')
    text = ""
    for paragraph in article.find_all('p'):
        text += paragraph.get_text() + "\n"

    # Process the text with spaCy
    doc = nlp(text)

    # Initialize variables to store content words and their frequencies
    content_words = defaultdict(int)

    # Define a set of allowed PoS tags for content words (nouns, adjectives)
    allowed_pos_tags = {"NOUN", "PROPN", "ADJ"}

    # Extract content words and their frequencies
    for token in doc:
        if token.pos_ in allowed_pos_tags:
            content_words[token.text] += 1

    # Sort content words by frequency in descending order
    sorted_content_words = sorted(content_words.items(), key=lambda x: x[1], reverse=True)

    # Create a summary by selecting the top content words
    selected_words = [word for word, _ in sorted_content_words[:summary_length]]
    summary = " ".join(selected_words)

    return text, summary

# Sample news article URL
article_url = "https://edition.cnn.com/2023/10/30/americas/asteroid-dust-dinosaur-extinction-photosynthesis-scn/index.html"

# Process and summarize the news article
article_text, article_summary = summarize_news_article(article_url, summary_length=10)

# Print the article text
print("Article Text:")
print(article_text)
print("\n")

# Print the summary
print("Summary:")
print(article_summary)
