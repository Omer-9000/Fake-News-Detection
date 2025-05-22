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
from sentence_transformers import SentenceTransformer


#Loading classifier
classifier=load(r'Fake-News-Detection\classifier_log.pkl')

#Loading scaler
scale=load(r'Fake-News-Detection\scaler.pkl')

#Load geo fact check articles embeddings
with open(r"Fake-News-Detection\news_vectors.pkl", "rb") as f:
    titles, links, title_vectors = pkl.load(f)


model = SentenceTransformer('Fake-News-Detection\minilm_model') 


#Labels
def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'True News'
  
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

    # Set a threshold
    threshold = 0.5

    # Get indices of scores above the threshold
    high_score_indices = [i for i, score in enumerate(scores) if score >= threshold]

    if not high_score_indices:
       print("No matching articles found!")
    else:
      for idx in high_score_indices:
          print(f"{scores[idx]:.2f} | {titles[idx]} → {links[idx]}")

#FUNCTION FOR PREDICTION
def prediction_func(news,lr_model):
  query = wordopt(news)
  query_vec = model.encode(query, convert_to_tensor=True)
  query_vec = np.array(query_vec).reshape(1, -1)
  query_vec=query_vec.tolist()
  query_vec=scale.transform(query_vec)
  pred_LR = lr_model.predict(query_vec)
  print("LR Prediction: {}".format(output_label(pred_LR[0])))
  
  print("\nChecking Geo Fact Check matches...")
  matches = check_fact(query_vec)

print("Enter the news to be checked: ")
news = str("Viral news of Pakistan mercilessly demolishing 6 indian jets.")
prediction_func(news,classifier)