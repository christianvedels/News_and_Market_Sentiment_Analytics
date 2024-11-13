# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:33:56 2024

@author: chris
"""

import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the ZipfIt function
def ZipfIt(textID):
    # Load and preprocess the text
    file_path = f'Coding_challenge_data/Texts/text{textID}.txt'
    with open(file_path, 'r', encoding="iso-8859-1") as f:
        text = f.read().lower()
        
    # Tokenize and count word frequencies
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    
    # Sort by frequency and rank the words
    freq = np.array(sorted(word_counts.values(), reverse=True))
    rank = np.arange(1, len(freq) + 1)
    
    # Log-transform the data
    log_rank = np.log(rank)
    log_freq = np.log(freq)
    
    # Perform linear regression
    slope, intercept, _, _, _ = linregress(log_rank, log_freq)
    
    return slope

if __name__ == "__main__":
    # Apply ZipfIt to each text and store results
    text_ids = range(1000)
    slopes = {textID: ZipfIt(textID) for textID in text_ids}
    
    # Plot the histogram of slopes
    plt.figure(figsize=(8, 5))
    plt.hist(slopes.values(), bins=30, edgecolor='black')
    plt.xlabel("Zipf Slope")
    plt.ylabel("Frequency")
    plt.title("Histogram of Zipf Slopes")
    plt.savefig("Coding_challenge_histogram.png")

    # Classify based on slope and test quality
    # Load answers for testing accuracy
    answers = pd.read_csv('Coding_challenge_data/Answer.csv')

    # Define a threshold for classification based on Zipf slope
    threshold = -0.8  # Adjust based on observed distribution of slopes (histogram)
    predictions = [x < threshold for x in slopes.values()]

    # Evaluate accuracy
    # Compare predictions to actual answers
    test = answers.Real == predictions 
    acc = np.mean(test)
    print(f'Accuracy: {acc:.2%}')

    
    
    
