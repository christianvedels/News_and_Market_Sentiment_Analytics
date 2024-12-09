from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer from HuggingFace
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Zero-shot classifier using cosine distance
def zero_shot_classifier(sentence, labels):
    sentence_embedding = get_embeddings(sentence)
    label_embeddings = [get_embeddings(label) for label in labels]
    similarities = [cosine_similarity(sentence_embedding, label_embedding) for label_embedding in label_embeddings]
    return labels[np.argmax(similarities)]

if __name__ == "__main__":
    # Sentences to test
    sentences = [
        "He banked the plane sharply to avoid the ridge.",
        "He was a master banker because he built the best banks in the trench.",
        "He was a master banker - all accounts were kept in order.",
        "After hearing the sentence, he felt a mix of relief and regret, knowing that justice had been served.",
        "This is just a regular old sentence."
    ]

    # Get embeddings for each sentence
    sentence_embeddings = [get_embeddings(sentence) for sentence in sentences]

    # Calculate mean of word embeddings for each sentence
    mean_word_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        mean_word_embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    # Compare mean of word embeddings with sentence embeddings
    similarities = []
    for i, sentence in enumerate(sentences):
        similarity = cosine_similarity(mean_word_embeddings[i], sentence_embeddings[i])
        similarities.append(similarity[0][0])
        print(f"Sentence: {sentence}")
        print(f"Cosine similarity: {similarity[0][0]}")

    # Plot the similarities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(len(sentences))), y=similarities)
    plt.xlabel('Sentence Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Mean Word Embeddings and Sentence Embeddings')
    plt.show()

    # Example usage of zero-shot classifier
    labels = ["finance", "aviation", "construction work", "legal", "general"]
    for i, sentence in enumerate(sentences):
        predicted_label = zero_shot_classifier(sentence, labels)
        print(f"Predicted label for '{sentence}': {predicted_label}")

    # Applying this zero shot classifier to Coding Challenge 1 is left as an exercise
    # https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%203%20-%20Classification%20pt%201/Coding_challenge_lecture3.md
