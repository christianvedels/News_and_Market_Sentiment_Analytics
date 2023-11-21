# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:07:13 2023

@author: chris
"""

#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# %% Libraries
from datetime import datetime
from newsapi import NewsApiClient
import spacy
from collections import Counter

# %% Params
end_date = datetime.now().strftime('%Y-%m-%d') # Today
start_date = '2023-11-14'

# %% Get news
# Notice that this borrows from Lecture 2 - Lexical resources/Code/01_Get_stock_data.py
# Get your own free API key here: https://newsapi.org/
# Init
newsapi = NewsApiClient(api_key="b973a0633202458792a421940fc2c1be")

# /v2/top-headlines
news = newsapi.get_everything(
    q='OpenAI',
    from_param=start_date,
    to=end_date,
    language='en'
    )

articles = news["articles"]
articles

# %% 
news_data = []
for news_item in articles:
    news_time = datetime.fromisoformat(news_item['publishedAt']).strftime('%Y-%m-%d %H:%M:%S')
    news_source = news_item['source']['name']
    news_headline = news_item['title']
    news_content = news_item['content']
    news_data.append([news_time, news_source, news_headline, news_content])
    
# %% Function to extract features
nlp = spacy.load("en_core_web_sm")

# Define function
def get_info(x):
    # Process the news headline with spaCy
    doc = nlp(x)

    # Extract Noun Phrases (NP) and Verb Phrases (VP) using spaCy's dependency parsing
    noun_phrases    = [chunk.text for chunk in doc.noun_chunks]
    verb_phrases    = [token.text for token in doc if token.pos_ == 'VERB']
    named_entities  = [(ent.text, ent.label_) for ent in doc.ents]
    organiations    = [org[0] for org in named_entities if org[1]=="ORG"]
    geography       = [gpe[0] for gpe in named_entities if gpe[1]=="GPE"]
    
    # Count the occurrences of each word
    noun_phrases    = sorted(Counter(noun_phrases).items(), key=lambda x: x[1], reverse=True)
    verb_phrases    = sorted(Counter(verb_phrases).items(), key=lambda x: x[1], reverse=True)
    named_entities  = sorted(Counter(named_entities).items(), key=lambda x: x[1], reverse=True)
    organiations    = sorted(Counter(organiations).items(), key=lambda x: x[1], reverse=True)
    geography       = sorted(Counter(geography).items(), key=lambda x: x[1], reverse=True)
    
    return noun_phrases, verb_phrases, named_entities, organiations, geography

# %% Apply
for x in news_data:
    noun_phrases, verb_phrases, named_entities, organiations, geography = get_info(x[3])
    
    print(f'\n--->Date: {x[0]}')
    print(f'Headline: {x[2]}')
    print(f'Geography: {geography}')
    print(f'Organisations: {organiations}')
    print(f'Verb phrases: {verb_phrases}')
