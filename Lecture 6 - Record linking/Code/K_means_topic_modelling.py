# -*- coding: utf-8 -*-
"""
@author: christian-vs
"""

# Topic Modeling with Bag of Words and K means

# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import gutenberg
import random
import numpy as np
import matplotlib.pyplot as plt
import spacy

def main():
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Step 0: Load and preprocess Gutenberg corpus
    paragraphs = [paragraph for file_id in gutenberg.fileids() for paragraph in gutenberg.paras(file_id)]
    random.shuffle(paragraphs)  # Shuffling for variety

    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Tokenize and lemmatize using spaCy
        doc = nlp(' '.join(paragraph[0]).lower())
        lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha]
        # Remove stop words
        filtered_tokens = [token for token in lemmatized_tokens if not nlp.vocab[token].is_stop]
        cleaned_paragraphs.append(' '.join(filtered_tokens))

    # Step 1: Create Bag of Words (BoW) representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned_paragraphs)

    # Step 2: Dimensionality reduction
    print(f'Shape before filtering: {X.shape}')  # Many columns
    # We will only use 90% of the words
    threshold = 0.90
    # Sum each column of X
    column_sums = X.sum(axis=0)

    # Convert to a numpy array for easier manipulation
    column_sums_array = column_sums.A1

    # Sort indices in descending order based on column sums
    sorted_indices = np.argsort(column_sums_array)[::-1]

    # Calculate the cumulative sum of column sums
    cumulative_sum = np.cumsum(column_sums_array[sorted_indices])

    # Find the index where cumulative sum exceeds 'threshold' of total tokens
    threshold_index = np.argmax(cumulative_sum >= threshold * cumulative_sum[-1])

    # Select only the top elements up to the threshold index
    selected_indices = sorted_indices[:threshold_index + 1]
    X_filtered = X[:, selected_indices]

    # Extract list of words in retained parts of X
    feature_names = np.array(vectorizer.get_feature_names_out())
    retained_tokens = feature_names[selected_indices].tolist()

    print(f'Shape after filtering: {X_filtered.shape}')  # Fewer columns

    # Step 3: Apply K-means
    # Initialize a list to store the RMSE values
    rmse_values = []

    # Iterate over different numbers of clusters
    for k in range(5, 21):  # Assuming you want to try clusters from 5 to 20
        print(f'fitting k = {k}')
        num_clusters = k

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=20, n_init=10)
        kmeans.fit(X_filtered)

        # Get cluster assignments for each paragraph
        cluster_assignments = kmeans.labels_

        # Compute distances from each point to its centroid
        distances = kmeans.transform(X_filtered)

        # Calculate the mean squared error (MSE)
        mse = np.mean(np.min(distances, axis=1)**2)

        # Calculate the RMSE
        rmse = np.sqrt(mse)

        # Append the RMSE value to the list
        rmse_values.append(rmse)

    # Identify the elbow point
    diffs = np.diff(rmse_values)
    elbow_point = np.where(diffs > 0)[0][0] + 5  # Add 5 because we started from k=5

    # Plot the RMSE values against the number of clusters
    plt.plot(range(5, 21), rmse_values, marker='x')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Root Mean Squared Error (RMSE)')

    # Mark the elbow point on the plot
    plt.scatter(elbow_point, rmse_values[elbow_point - 5], c='red', label='Elbow Point', marker='o', s=50)
    plt.legend()

    plt.show()

    # Print the identified optimal number of clusters
    print(f"Optimal Number of Clusters (Elbow Point): {elbow_point}")

    # Refit with optimal clusters
    kmeans = KMeans(n_clusters=elbow_point, random_state=20, n_init=10)
    kmeans.fit(X_filtered)

    # Look at the results
    # Does the topics make sense?
    # Get cluster assignments for each paragraph
    cluster_assignments = kmeans.labels_

    # Turn everything into frequency dictionaries
    dicts_of_freqs = []
    for cluster_id in range(elbow_point):
        cluster_indices = (cluster_assignments == cluster_id)

        # Extract the rows corresponding to the cluster from X_filtered
        cluster_X = X_filtered[cluster_indices]

        # Sum the columns to get the total frequency for each word
        word_frequencies = np.array(cluster_X.sum(axis=0)).reshape(-1)

        # Get the feature names (words) from the vectorizer
        feature_names = np.array(vectorizer.get_feature_names_out())[selected_indices]

        # Create a dictionary of word frequencies
        word_frequency_dict = dict(zip(feature_names, word_frequencies))

        dicts_of_freqs.append(word_frequency_dict)

    # Which words with freq>1 appear in all dictionaries?
    common_words = set.intersection(*[
        set(word for word, freq in word_frequency_dict.items() if freq > 1)
        for word_frequency_dict in dicts_of_freqs
    ])

    i = 0
    print("Top 10 words for each cluster:")
    for dict_i in dicts_of_freqs:
        # Sort the dictionary by values in descending order
        sorted_word_frequency = dict(sorted(dict_i.items(), key=lambda item: item[1], reverse=True))

        # Filter away common words
        filtered_word_frequency = {k: v for k, v in sorted_word_frequency.items() if k not in common_words}

        # Display only the top 10 word frequencies
        top_10_word_frequency = {k: sorted_word_frequency[k] for k in list(sorted_word_frequency)[:10]}

        print(f"Cluster {i+1}:\nTop 10 Word Frequencies: {top_10_word_frequency}\n{'-'*50}")
        i = i + 1

if __name__ == "__main__":
    main()
