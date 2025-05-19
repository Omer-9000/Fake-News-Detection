import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
from joblib import load
import pickle as pkl
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

classifier=load(r'linear_model.pkl')
vectorize=load(r'vectorizer.pkl')

# Start browser
'''driver = webdriver.Chrome()
driver.get("https://www.geo.tv/category/geo-fact-check")


# Scroll to bottom to trigger JS to load more content
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    try:
        load_button = driver.find_element(By.XPATH, "//button[contains(text(),'Load More')]")
        load_button.click()
        time.sleep(5)
    except:
        break

# Get full rendered HTML
html = driver.page_source
'''

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
  articles = soup.find_all("div",class_ = "col-sm-6 col-lg-4")
  print(len(articles))
  matches = []

  for article in articles:
    try:
      a_tag = article.find(class_="text-body")
      title = a_tag.find("img")
      title=title['alt']
      
      href = a_tag['href']
      score = similarity(news, title)
      if score > 0.5:
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
      print(f"Match (Score {score:.2f}):\n {title}\n {url}")
  else:
    print("No matching articles found on Geo Fact Check.")


print("Enter the news to be checked: ")
news = str("document claims radiation leak in pakistan")
prediction_func(news,classifier,vectorize)
