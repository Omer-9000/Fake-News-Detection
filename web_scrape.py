import pandas as pd
import numpy as np
import csv
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
from selenium.webdriver.common.action_chains import ActionChains
import time


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


driver = webdriver.Chrome()
driver.get("https://www.geo.tv/category/geo-fact-check")
wait = WebDriverWait(driver, 10)
load_more_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "load_more_fact_news")))

while load_more_button:
  ActionChains(driver).move_to_element(load_more_button).perform()
  load_more_button.click()
  time.sleep(2)
  try:
    load_more_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "load_more_fact_news")))
  except:
    break

html_source = driver.page_source

soup = BeautifulSoup(html_source, 'html.parser')
articles = soup.find_all("div",class_ = "col-sm-6 col-lg-4")
print(len(articles))


#Saving the news and links in csv file
with open(r"Fake-News-Detection\geo_articles.csv", "w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["title", "link"])
    for article in articles:
      a_tag = article.find(class_="text-body")
      title = a_tag.find("img")
      title=title['alt']
      link = a_tag['href']
      writer.writerow([title, link])

print("Saved articles to geo_articles.csv")


#Loading the csv file
titles=[]
links=[]
with open(r"Fake-News-Detection\geo_articles.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        titles.append(row["title"])
        links.append(row["link"])

title_vectors = [sentence_embedding(title, glove) for title in titles]

# Save for later use
with open("Fake-News-Detection\glove_title_vectors.pkl", "wb") as f:
    pkl.dump((titles, links, title_vectors), f)


