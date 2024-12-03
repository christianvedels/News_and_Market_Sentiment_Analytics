# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:40:53 2023

@author: christian-vs
"""

# Topic Modeling with Latent Dirichlet Allocation (LDA) on Reuters dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import reuters
import spacy
import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load spacy model
    nlp = spacy.load("en_core_web_sm")

    # Step 0: Load and preprocess Reuters dataset
    docs = [reuters.raw(file_id) for file_id in reuters.fileids()]
    random.shuffle(docs)  # Shuffling for variety

    cleaned_docs = []
    for doc in docs:
        # Process document with spacy
        spacy_doc = nlp(doc.lower())
        # Tokenize, lemmatize, and remove stop words
        tokens = [token.lemma_ for token in spacy_doc if token.is_alpha and not token.is_stop]
        cleaned_docs.append(' '.join(tokens))
        
    # Step 1: Create Bag of Words (BoW) representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned_docs)

    # Step 2: Dimensionality reduction
    print(f'Shape after filtering: {X.shape}') # Many columns
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

    print(f'Shape after filtering: {X_filtered.shape}') # Fewer columns

    # Step 3: Determine the optimal number of topics using an elbow point
    eval_metric_values = []
    num_topics_range = range(5, 21)  # Assuming you want to try topics from 5 to 20

    for num_topics in num_topics_range:
        print(f'fitting num_topics = {num_topics}')
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=20)
        lda.fit(X_filtered)
        
        # Evaluation metric (you can replace this with your preferred metric)
        perplexity = lda.perplexity(X_filtered)
        
        eval_metric_values.append(perplexity)

    # Identify the elbow point
    diffs = np.diff(eval_metric_values)
    elbow_point = np.where(diffs > 0)[0][0] + 5  # Add 5 because we started from num_topics=5

    # Plot the evaluation metric values against the number of topics
    plt.plot(num_topics_range, eval_metric_values, marker='x')
    plt.title('Elbow Method for Optimal Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')  # Replace 'Perplexity' with your chosen metric

    # Mark the elbow point on the plot
    plt.scatter(elbow_point, eval_metric_values[elbow_point - 5], c='red', label='Elbow Point', marker='o', s=50)
    plt.legend()

    plt.show()

    # Print the identified optimal number of topics
    print(f"Optimal Number of Topics (Elbow Point): {elbow_point}")

    # Refit with optimal topics
    elbow_point = 8
    lda = LatentDirichletAllocation(n_components=elbow_point, random_state=20)
    lda.fit(X_filtered)

    # Look at the results
    # Display the top words for each topic
    feature_names = np.array(vectorizer.get_feature_names_out())[selected_indices]

    print("Top words for each topic:")
    for topic_id, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-11:-1]  # Get the indices of the top 10 words
        top_words = feature_names[top_word_indices]
        print(f"Topic {topic_id + 1}: {', '.join(top_words)}\n{'-'*50}")

if __name__ == "__main__":
    main()
