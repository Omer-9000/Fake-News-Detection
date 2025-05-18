import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
from joblib import load
import pickle as pkl

classifier=load(r'linear_model.pkl')
vectorize=load(r'vectorizer.pkl')

def similarity(a,b):
  return SequenceMatcher(None,a.lower(),b.lower()).ratio()

def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'Not Fake News'
  
def wordopt(text):
  text = str(text).lower()
  text = re.sub('\[.*?/]','',text)
  text = re.sub("\\W"," ",text)
  text = re.sub('https?://\S+|www\.\S+','',text)
  text = re.sub('<.*?>+','',text)
  text = re.sub('[%s]'%re.escape(string.punctuation),'',text)
  text = re.sub('\n','',text)
  text = re.sub('\w*\d\w*','',text)
  return text
  
#WEB SCRAPING
def check_fact(news):
  url = "https://www.geo.tv/category/geo-fact-check"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  articles = soup.find_all("div", class_ = "listing")
  matches = []

  for article in articles:
    try:
      a_tag = article.find("a")
      title = a_tag.text.strip()
      href = "https://www.geo.tv" + a_tag['href']
      score = similarity(news, title)
      if score > 0.2:
        matches.append((title, href, score))
    except:
      continue
  matches = sorted(matches, reverse = True)
  return matches[:3]


#FUNCTION FOR PREDICTION

def prediction_func(news,lr_model,vect):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test['text'].apply(wordopt)
  new_x_test = new_def_test["text"]
  new_xv_test = vect.transform(new_x_test)
  pred_LR = lr_model.predict(new_xv_test)
  print("\nLR Prediction: {}".format(output_label(pred_LR[0])))
  
  print("\nChecking Geo Fact Check matches...")
  matches = check_fact(news)
  if matches:
    for title, url, score in matches:
      print(f"Match (Score {score:.2f}): {title} — {url}")
  else:
    print("No matching articles found on Geo Fact Check.")


print("Enter the news to be checked: ")
news = str("Viral 'Pakistani Pilot' photo is actually from Turkey")
prediction_func(news,classifier,vectorize)
