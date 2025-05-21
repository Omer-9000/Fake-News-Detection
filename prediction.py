import pandas as pd
import numpy as np
import csv
import re
import string
from bs4 import BeautifulSoup
import requests
from joblib import load
import pickle as pkl
import time
from sklearn.metrics.pairwise import cosine_similarity


#Loading classifier
classifier=load(r'Fake-News-Detection\classifier_mod.pkl')

#Load geo fact check articles embeddings
with open(r"Fake-News-Detection\glove_title_vectors.pkl", "rb") as f:
    titles, links, title_vectors = pkl.load(f)

#Load Glove
with open(r"Fake-News-Detection\glove_100d.pkl", "rb") as f:
    glove = pkl.load(f)

#Embeddings function
def sentence_embedding(sentence, data_dit,dim=100):
  words = sentence.lower().split()
  word_embeddings = []
  for word in words:
    if word in data_dit:
      word_embeddings.append(data_dit[word])
  if not word_embeddings:
    return np.zeros(dim)

  sentence_embed = np.mean(word_embeddings, axis=0)
  return sentence_embed

#Labels
def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'Not Fake News'
  
#Preprocessing function
def wordopt(text):
  text = str(text).lower()
  text = re.sub(r'\[.*?\]', '', text)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
  text = re.sub(r'\n', ' ', text) 
  text = re.sub(r'\w*\d\w*', '', text) 
  text = re.sub(r'\s+', ' ', text).strip() 
  return text

#WEB SCRAPING
def check_fact(query):
    # Compare with cosine similarity
    scores = cosine_similarity(query, title_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    # Show results
    for idx in top_indices:
        print(f"{scores[idx]:.2f} | {titles[idx]} → {links[idx]}")

#FUNCTION FOR PREDICTION
def prediction_func(news,lr_model):
  query = wordopt(news)
  query_vec = sentence_embedding(query, glove)
  query_vec = np.array(query_vec).reshape(1, -1)
  query_vec=query_vec.tolist()
  pred_LR = lr_model.predict(query_vec)
  print("LR Prediction: {}".format(output_label(pred_LR[0])))
  
  print("\nChecking Geo Fact Check matches...")
  matches = check_fact(query_vec)

print("Enter the news to be checked: ")
news = str("Viral news of Pakistani pilot in indian custody spreading all over the country.")
prediction_func(news,classifier)