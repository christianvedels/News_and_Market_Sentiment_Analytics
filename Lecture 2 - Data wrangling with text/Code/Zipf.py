import spacy
from collections import Counter
import requests
import matplotlib.pyplot as plt

# Define a function to calculate word frequencies using spaCy
def get_word_frequencies(text):
    """
    Process text using spaCy and calculate word frequencies.
    
    Parameters:
    text (str): The input text to analyze.

    Returns:
    Counter: A Counter object containing word frequencies.
    """
    doc = nlp(text)
    # Extract words, convert to lowercase, and filter out punctuation and stop words
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return Counter(words)

# Function to get Alice in Wonderland
def get_alice_in_wonderland():
    url = "https://www.gutenberg.org/cache/epub/19033/pg19033.txt"
    response = requests.get(url)
    text = response.text
    return text

if __name__ == "__main__":
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Load the text of "Alice in Wonderland" from Project Gutenberg
    text = get_alice_in_wonderland()

    # Get word frequencies using the defined function
    word_freq = get_word_frequencies(text)
    
    # Sort words by frequency
    sorted_freq = word_freq.most_common()
    
    # Separate ranks and frequencies for plotting
    ranks = range(1, len(sorted_freq) + 1)
    frequencies = [freq for _, freq in sorted_freq]
    
    # Plot the Zipf distribution
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank of word (log scale)")
    plt.ylabel("Frequency of word (log scale)")
    plt.title("Zipf's Law Distribution of Words in 'Alice in Wonderland'")
    plt.savefig("Lecture 2 - Data wrangling with text/Code/zipf_distribution.png")

    # Point: There is a curious relationship between rank and frequency in natural language texts.
