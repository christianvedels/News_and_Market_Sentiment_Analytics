# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:17:01 2023

@author: christian-vs
"""

import spacy
from collections import defaultdict

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Sample text (news article)
text = """
In a breakthrough discovery, scientists have found evidence of water on Mars.
The presence of water on the red planet has significant implications for the possibility of life.
This discovery was made using advanced technology and telescopes.
"""

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
summary_length = 10  # Number of content words to include in the summary
summary = " ".join(word for word, _ in sorted_content_words[:summary_length])

# Print the summary
print("Summary:")
print(summary)
