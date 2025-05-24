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

#Loading geo fact check articles embeddings
with open(r"Fake-News-Detection\news_vectors.pkl", "rb") as f:
    titles, links, title_vectors = pkl.load(f)


model = SentenceTransformer('Fake-News-Detection\minilm_model') 


#Labels
def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'True News'
  

#FIND SIMILAR ARTICLES
def check_fact(query,model,scaler):
    query_vec = model.encode(query, convert_to_tensor=True)
    query_vec = np.array(query_vec).reshape(1, -1)
    query_vec=query_vec.tolist()

    #Set a threshold
    threshold = 0.5
    #Compare with cosine similarity
    scores = cosine_similarity(query_vec, title_vectors)[0]
    #Pair each score with its original index
    indexed_scores = list(enumerate(scores))

    #Sort the (index, score) pairs by score in descending order
    sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    #Filter top scores above threshold
    high_score_results = [(score, titles[i], links[i]) for i, score in sorted_indexed_scores if score >= threshold]

    #Return top 3 results
    return high_score_results[:3] if high_score_results else None

#FUNCTION FOR PREDICTION
def prediction_func(q,lr_model,model,scaler):
  query = q
  query_vec = model.encode(query, convert_to_tensor=True)
  query_vec = np.array(query_vec).reshape(1, -1)
  query_vec=query_vec.tolist()
  query_vec=scaler.transform(query_vec)
  pred_LR = lr_model.predict(query_vec)
  return pred_LR[0]  