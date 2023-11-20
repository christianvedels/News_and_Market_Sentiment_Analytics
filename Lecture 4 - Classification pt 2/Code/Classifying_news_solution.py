# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:33:39 2023

@author: christian-vs
"""
# %%
import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Additions
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import wordnet as wn

# %%
# Download NLTK resources if not already downloaded
nltk.download('reuters')
nltk.download('stopwords')

# Load Reuters articles and categories
documents = reuters.fileids()
categories = reuters.categories()

# %% Function to extract word type as a feature
def word_type_feature(word):
    pos_tag = nltk.pos_tag([word])[0][1]
    return {'word_type': pos_tag}

# %% Lemmatizer
def lemmatizer(word):
    res = wn.synsets(word)[0]
    return wn.synsets(word)[0]

lemmatizer("Word")

# %% Prepare data
# Preprocess and prepare the data
data = []
labels = []

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

for doc_id in documents:
    # print(reuters.raw(doc_id))
    # words = [stemmer.stem(word.lower()) for word in word_tokenize(reuters.raw(doc_id))]
    # filtered_words = [word for word in words if word not in stop_words]
    # data.append(' '.join(filtered_words))
    words = [stemmer.stem(word.lower()) for word in word_tokenize(reuters.raw(doc_id))]
    filtered_words = [word for word in words if word not in stop_words]
    lemmas = [lemmatizer(word) for word in filtered_words]
    word_type = [word_type_feature(word) for word in filtered_words]
    data.append(' '.join(filtered_words) + ' '.join(word_type))
    labels.append(reuters.categories(doc_id))

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_data = tfidf_vectorizer.fit_transform(data)

# Convert multi-label categories into a binary matrix
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_data, binary_labels, test_size=0.2, random_state=20)

# %% Train a Naive Bayes classifier
classifier = MultiOutputClassifier(MultinomialNB())
classifier.fit(X_train, y_train)

classifier_rf = RandomForestClassifier()
classifier_rf.fit(X_train, y_train)

# %% Make predictions
# y_pred = classifier.predict(X_test)
y_pred = classifier_rf.predict(X_test)

# %% Evaluate the model accuracy
# Precision:  If I guess x, how often is that guess then correct?
# Recall:     If x is correct, how often will I then guess it?
# F1:         Geometric mean of the two  
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# %% Print classification report
category_names = mlb.classes_
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=category_names))

