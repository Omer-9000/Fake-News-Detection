import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher

#datasets
data_fake = pd.read_csv('fakified_news.csv')
data_true = pd.read_csv('Hopefully true.csv')

#head
#print(data_fake.head())
#print(data_true.head())

data_fake["class"] = 0
data_true["class"] = 1

'''
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
  data_fake.drop([i],axis=0,inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
  data_true.drop([i],axis=0,inplace=True)

#print(data_fake.shape,data_true.shape)


data_fake_manual_testing = data_fake.tail(10).copy()
data_true_manual_testing = data_true.tail(10).copy()

data_fake_manual_testing.loc[:, 'class'] = 0
data_true_manual_testing.loc[:, 'class'] = 1

'''

data_merge = pd.concat([data_fake,data_true],axis = 0)
data = data_merge.drop(['title'],axis = 1)

#shuffling
data = data.sample(frac = 1)

data.reset_index(inplace = True)
data.drop(['index'],axis = 1, inplace = True)

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

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)



vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)



LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))

DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt = DT.predict(xv_test)
print(pred_dt)
print(DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train,y_train)
predict_gb = GB.predict(xv_test)
print(predict_gb)

print(GB.score(xv_test, y_test))
print(classification_report(y_test,predict_gb))

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train,y_train)
predict_rf = RF.predict(xv_test)
print(predict_rf)
print(RF.score(xv_test, y_test))
print(classification_report(y_test, predict_rf))

def similarity(a,b):
  return SequenceMatcher(None,a.lower(),b.lower()).ratio()

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


def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'Not Fake News'

def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test['text'].apply(wordopt)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorization.transform(new_x_test)
  pred_LR = LR.predict(new_xv_test)
  pred_DT = DT.predict(new_xv_test)
  pred_GB = GB.predict(new_xv_test)
  pred_RF = RF.predict(new_xv_test)
  print("\nLR Prediction: {}".format(output_label(pred_LR[0])))
  print("DT Prediction: {}".format(output_label(pred_DT[0])))
  print("GB Prediction: {}".format(output_label(pred_GB[0])))
  print("RF Prediction: {}".format(output_label(pred_RF[0])))
  
  print("\nChecking Geo Fact Check matches...")
  matches = check_fact(news)
  if matches:
    for title, url, score in matches:
      print(f"Match (Score {score:.2f}): {title} — {url}")
  else:
    print("No matching articles found on Geo Fact Check.")


print("Enter the news to be checked: ")
news = str("Viral 'Pakistani Pilot' photo is actually from Turkey")
manual_testing(news)
