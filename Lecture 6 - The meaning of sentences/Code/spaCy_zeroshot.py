# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:56:10 2023

@author: chris
"""

#%%
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pandas as pd

# %% Plotly stuff
import plotly.express as px
import plotly.graph_objects as go

# Show in browser
import plotly.io as pio
pio.renderers.default='browser'

#%%
# Load spaCy model
import spacy_transformers
nlp = spacy.load("en_core_web_trf")

# %%
# Function to get spaCy embeddings for a sentence
def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    return doc.vector

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vector1, vector2):
    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

# Function for zero-shot emotion classification
def classify_emotion(sentence, emotion_classes):
    sentence_embedding = get_sentence_embedding(sentence)
    similarities = [calculate_cosine_similarity(sentence_embedding, get_sentence_embedding(class_sentence)) for class_sentence in emotion_classes]
    predicted_class = emotion_classes[np.argmax(similarities)]
    return predicted_class

#%%
# 88 examples of sentences for emotion classification
example_sentences = [
    "I am feeling so happy today!",
    "Wow! That was unexpected!",
    "I can't believe you did that! It makes me furious.",
    "I feel really down and sad right now.",
    "I'm overjoyed with excitement!",
    "This news caught me off guard.",
    "Your actions infuriate me!",
    "Life feels really tough and depressing at the moment.",
    "I'm thrilled about the good news!",
    "Surprise! I didn't see that coming.",
    "I'm so angry about the situation.",
    "Feeling a bit gloomy and blue today.",
    "Pure joy and happiness overwhelm me.",
    "What a shock! I didn't anticipate this.",
    "I'm enraged by your behavior.",
    "The news is heartbreaking, and I feel sad.",
    "My heart is filled with joy and delight.",
    "I'm taken aback by this surprising revelation.",
    "I'm absolutely furious right now.",
    "Feeling a sense of melancholy and sorrow.",
    "I'm on cloud nine with happiness!",
    "This unexpected turn of events is astonishing.",
    "Your actions make me incredibly mad.",
    "A wave of sadness washes over me.",
    "I'm delighted beyond words!",
    "I'm utterly surprised by what just happened.",
    "I'm seething with anger!",
    "Feeling a deep sense of sadness.",
    "I'm so happy I could dance!",
    "I never saw that coming. What a surprise!",
    "Your behavior infuriates me.",
    "Feeling a bit down and low-spirited.",
    "Pure bliss and happiness fill my heart.",
    "The news caught me by surprise.",
    "I'm boiling with anger!",
    "Experiencing a profound sense of sadness.",
    "I'm thrilled to bits!",
    "I'm shocked by the unexpected turn of events.",
    "Your actions make my blood boil!",
    "Feeling a heavy heart and sadness.",
    "Ecstatic and overjoyed with happiness!",
    "This surprising news has left me speechless.",
    "I'm absolutely livid right now.",
    "Feeling a sense of sorrow and unhappiness.",
    "I'm elated beyond measure!",
    "The unexpected news has left me in awe.",
    "I'm furious about what just happened.",
    "Experiencing a deep and overwhelming sadness.",
    "I'm on cloud nine with pure joy!",
    "This unexpected revelation is truly astonishing.",
    "Your actions have left me fuming!",
    "Feeling a profound sadness and heartache.",
    "I'm over the moon with happiness!",
    "The surprising twist in the story is mind-blowing.",
    "I'm boiling with anger over your behavior.",
    "A wave of sadness engulfs me.",
    "I'm absolutely delighted!",
    "I'm completely taken aback by this surprise.",
    "Your actions make me seethe with anger.",
    "Feeling a heavy heart and a deep sadness.",
    "I'm ecstatic and filled with joy!",
    "This unexpected turn of events is unbelievable.",
    "I'm fuming with anger right now.",
    "Experiencing a profound sense of sadness and grief.",
    "I'm overjoyed and thrilled!",
    "The surprising news has left me in shock.",
    "I'm absolutely furious about what just happened.",
    "Feeling a deep sadness and a heavy heart.",
    "I'm on cloud nine with happiness and bliss!",
    "The unexpected revelation is truly astonishing.",
    "Your actions make me absolutely livid.",
    "Experiencing a profound sadness and heartache.",
    "I'm elated and filled with pure joy!",
    "The surprising twist in the story is mind-blowing.",
    "I'm boiling with anger over your behavior.",
    "A wave of sadness engulfs me.",
    "I'm absolutely delighted!",
    "I'm completely taken aback by this surprise.",
    "Your actions make me seethe with anger.",
    "Feeling a heavy heart and a deep sadness.",
    "I'm ecstatic and filled with joy!",
    "This unexpected turn of events is unbelievable.",
    "I'm fuming with anger right now.",
    "Experiencing a profound sense of sadness and grief.",
    "I'm overjoyed and thrilled!",
    "The surprising news has left me in shock.",
    "I'm absolutely furious about what just happened.",
    "Feeling a deep sadness and a heavy heart."
]

# %%
# Emotion class labels
emotion_classes = ["Happy", "Surprised", "Angry", "Sad"]

# Generate embeddings for each example sentence
sentence_embeddings = [get_sentence_embedding(sentence) for sentence in example_sentences]

# Zero-shot emotion classification for each sentence
predictions = [classify_emotion(sentence, emotion_classes) for sentence in example_sentences]

for i in range(5): # Print a few
    print(f'{predictions[i]}: {example_sentences[i]}')

# Create a DataFrame to keep track of the order
df = pd.DataFrame({'Sentence': example_sentences, 'Prediction': predictions})

