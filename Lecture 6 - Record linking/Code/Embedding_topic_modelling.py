# -*- coding: utf-8 -*-
"""
@author: christian-vs
"""

# Topic Modeling with DistilBERT Embeddings and K means

# Import necessary libraries
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

def main():
    # Step 0: Load and preprocess Gutenberg corpus
    paragraphs = [paragraph for file_id in gutenberg.fileids() for paragraph in gutenberg.paras(file_id)]
    random.shuffle(paragraphs)  # Shuffling for variety
    paragraphs = paragraphs[0:5000] # Toydata

    paragraphs = [' '.join(x[0]) for x in paragraphs]

    # Step 1: Obtain DistilBERT sentence embeddings
    sent_encoder = SentenceTransformer('distilbert-base-uncased')
    X_embeddings = sent_encoder.encode(paragraphs, show_progress_bar=True)

    # Note: X_embeddings is now a matrix where each row corresponds to the DistilBERT embedding of a paragraph.

    # Step 2: Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=3, random_state=20)
    X_tsne = tsne.fit_transform(X_embeddings)

    # Step 3: Apply K-Means clustering with cosine distance
    # Initialize a list to store the RMSE values
    rmse_values = []

    # Iterate over different numbers of clusters
    for k in range(5, 21):  # Assuming you want to try clusters from 5 to 20
        print(f'fitting k = {k}')
        num_clusters = k

        # Apply K-Means clustering with cosine distance
        kmeans = KMeans(n_clusters=num_clusters, random_state=20, n_init = 10)
        kmeans.fit(X_tsne)

        # Get cluster assignments for each paragraph
        cluster_assignments = kmeans.labels_

        # Compute distances from each point to its centroid
        distances = cosine_distances(X_tsne, kmeans.cluster_centers_)

        # Calculate the mean squared error (MSE)
        mse = np.mean(np.min(distances, axis=1)**2)

        # Calculate the RMSE
        rmse = np.sqrt(mse)

        # Append the RMSE value to the list
        rmse_values.append(rmse)

    # Identify the elbow point
    diffs = np.diff(rmse_values)
    elbow_point = np.where(diffs > 0)[0][0] + 5  # Add 2 because we started from k=2

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
    elbow_point = 4
    kmeans = KMeans(n_clusters=elbow_point, random_state=20, n_init = 10)
    kmeans.fit(X_tsne)

    # Find central sentence
    central_sentences = []

    for cluster_id in range(elbow_point):  # Use the optimal number of clusters

        # Extract the rows corresponding to the cluster from X_filtered
        cluster_indices = (cluster_assignments == cluster_id)
        cluster_X = np.array(paragraphs)[cluster_indices]
        X_tsne_cluster = X_tsne[cluster_indices] 
        
        if len(cluster_X) > 0:
            # Find the paragraph index of the centroid (closest to the centroid)
            centroid_index = np.argsort(np.sum((X_tsne_cluster - kmeans.cluster_centers_[cluster_id])**2, axis=1))[:3]

            # Get the entire paragraph corresponding to the centroid
            central_sentence = cluster_X[centroid_index]

            central_sentences.append(central_sentence)

    # Look at the results
    # Does the topics make sense?
    # Get cluster assignments for each paragraph
    cluster_assignments = kmeans.labels_

    # Visualize clusters in 2D
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_assignments, cmap='viridis', alpha=0.5)
    plt.legend()
    plt.title('Clusters in t-SNE Reduced Space')
    plt.show()

    # Print central sentences
    print("\nCentral Sentences:")
    for i, sentence in enumerate(central_sentences):
        print(f"Cluster {i + 1}: {sentence}\n{'-' * 50}")

if __name__ == "__main__":
    main()