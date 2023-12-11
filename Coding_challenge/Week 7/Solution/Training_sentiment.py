# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 00:00:19 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

#%% Libraries
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% Load IMDb dataset from CSV
imdb_data_path = '../../../../IMDB/IMDB Dataset.csv'
imdb_df = pd.read_csv(
    imdb_data_path
    # , nrows = 2000 # Comment out for toyrun
    )

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(imdb_df, test_size=0.2, random_state=20)

# %% Data prep
# Tokenize and encode the training and validation sets
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_inputs = tokenizer(train_df['review'].tolist(), return_tensors='pt', padding=True, truncation=True)
val_inputs = tokenizer(val_df['review'].tolist(), return_tensors='pt', padding=True, truncation=True)

# Encode labels
label_encoder = LabelEncoder()
train_labels = torch.tensor(label_encoder.fit_transform(train_df['sentiment']), dtype=torch.long)
val_labels = torch.tensor(label_encoder.transform(val_df['sentiment']), dtype=torch.long)

# %% Create a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, inputs, labels, texts):
        self.inputs = inputs
        self.labels = labels
        self.texts = texts  # Add texts as an attribute

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'inputs': {key: tensor[idx] for key, tensor in self.inputs.items()},
            'labels': self.labels[idx],
            'texts': self.texts[idx]  # Add texts to the output
        }

# Create DataLoader for training and validation sets
train_dataset = SentimentDataset(train_inputs, train_labels, train_df['review'].tolist())
val_dataset = SentimentDataset(val_inputs, val_labels, val_df['review'].tolist())

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# %% Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)

# %% Training loop
# num_epochs = 2
# for epoch in range(num_epochs): # Appr. 15 min. per epoch
#     model.train()
#     for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#         inputs, labels, texts = batch['inputs'], batch['labels'], batch['texts'] # Modified this
#         inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(**inputs, labels=labels) # Equivalent to model(input_ids=tensor1, attention_mask=tensor2, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     correct_preds = 0
#     total_preds = 0
#     with torch.no_grad():
#         for batch in tqdm(val_dataloader, desc='Validation'):
#             inputs, labels, texts = batch['inputs'], batch['labels'], batch['texts'] # Modified this
#             inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
#             labels = labels.to(device)

#             outputs = model(**inputs, labels=labels)
#             loss = outputs.loss
#             val_loss += loss.item()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             correct_preds += (preds == labels).sum().item()
#             total_preds += labels.size(0)

#     avg_val_loss = val_loss / len(val_dataloader)
#     accuracy = correct_preds / total_preds

#     print(f'Epoch {epoch + 1}/{num_epochs}:')
#     print(f'  Training Loss: {loss.item():.4f}')
#     print(f'  Validation Loss: {avg_val_loss:.4f}')
#     print(f'  Validation Accuracy: {accuracy * 100:.2f}%')

# %% Save the fine-tuned model
# model.save_pretrained('fine_tuned_sentiment_model')
# tokenizer.save_pretrained('fine_tuned_sentiment_model')

# %% Load model
model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_sentiment_model')
tokenizer = DistilBertTokenizer.from_pretrained('fine_tuned_sentiment_model')

# %% Evaluate the model
# model.eval()
# all_preds = []
# all_labels = []

# with torch.no_grad():
#     for batch in tqdm(val_dataloader, desc='Validation'):
#         inputs, labels, texts = batch['inputs'], batch['labels'], batch['texts'] # Modified this
#         inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
#         labels = labels.to(device)

#         # Ensure model is on the same device as the inputs
#         model.to(inputs[list(inputs.keys())[0]].device)

#         outputs = model(**inputs)
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=1)

#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# %% Calculate and print classification report
# print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# %% Generate and plot confusion matrix
# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# # Normalize confusion matrix
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# # Generate and plot normalized confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title('Normalized Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

# %% Add NER
import spacy

nlp = spacy.load("en_core_web_sm")

# Define NER pipeline
def ner_pipeline(text):
    doc = nlp(text)
    
    # Extract named entities from the document
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities

# %% Run evaluation with NER
model.eval()
all_preds = []
all_labels = []

entity_sentiments = {}  # Dictionary to store entity sentiments

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc='Validation + entities'):
        inputs, labels, texts = batch['inputs'], batch['labels'], batch['texts']  # Modified this
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        labels = labels.to(device)

        # Ensure model is on the same device as the inputs
        model.to(inputs[list(inputs.keys())[0]].device)

        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Assuming sentiment values are obtained from logits
        sentiments = torch.softmax(logits, dim=1)[:, 1] - torch.softmax(logits, dim=1)[:, 0]

        # Get named entities and their sentiments
        reviews = texts
        for i, review in enumerate(reviews):
            entities = ner_pipeline(review)
            for entity in entities:
                text = entity[0]  # Assuming the entity is a tuple with (word, label)
                sentiment = sentiments[i].item()  # Assuming sentiments is a scalar value
                ent_type = entity[1]
                
                if text not in entity_sentiments:
                    entity_sentiments[text] = {'count': 1, 'sentiment_sum': sentiment, 'ent_type': ent_type}
                else:
                    entity_sentiments[text]['count'] += 1
                    entity_sentiments[text]['sentiment_sum'] += sentiment
                    entity_sentiments[text]['ent_type'] = ent_type # If different sentiment types, then overwrite

# %% Report results
# Calculate average sentiment for each entity
entity_avg_sentiments = {text: data['sentiment_sum'] / data['count'] for text, data in entity_sentiments.items() if data['count']>30}

# Print entities with the most negative and most positive average sentiment
sorted_entities = sorted(entity_avg_sentiments.items(), key=lambda x: x[1])
print("Top 10 Entities with Most Negative Average Sentiment:")
for entity, avg_sentiment in sorted_entities[:5]:
    print(f"{entity}: {avg_sentiment:.4f}")

print("\nTop 10 Entities with Most Positive Average Sentiment:")
for entity, avg_sentiment in sorted_entities[-5:]:
    print(f"{entity}: {avg_sentiment:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Split by type
all_types = [data['ent_type'] for text, data in entity_sentiments.items() if data['count']>30]
sentiments = [data['sentiment_sum']/data['count'] for text, data in entity_sentiments.items() if data['count']>30]
text = [text for text, data in entity_sentiments.items() if data['count']>30]
all_types = set(all_types)

for t in all_types:
    entity_avg_sentiments = {
        text+" ("+str(data['count'])+")": data['sentiment_sum'] / data['count'] 
        for text, data in entity_sentiments.items() 
        if data['ent_type']==t and data['count']>30
        }
    sorted_entities = sorted(entity_avg_sentiments.items(), key=lambda x: x[1])
    
    # Print
    print("===================")
    print(f"\nENTITY TYPE: {t}")
    print("Top 10 Entities with Most Negative Average Sentiment:")
    for entity, avg_sentiment in sorted_entities[:5]:
        print(f"{entity}: {avg_sentiment:.4f}")

    print("\nTop 10 Entities with Most Positive Average Sentiment:")
    for entity, avg_sentiment in sorted_entities[-5:]:
        print(f"{entity}: {avg_sentiment:.4f}")

