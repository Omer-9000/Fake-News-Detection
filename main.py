import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
from joblib import dump
import pickle as pkl


#Open Glove file
with open("Fake-News-Detection\glove.6B.100d.txt") as file:
    data = file.readlines()

#Convert Embeddings to a dictionary
data_dict = dict()
for i in range(len(data)):
    split_data = data[i].split()
    try:
      data_dict[split_data[0]] = np.array(split_data[1:]).astype('float64')
    except:
      pass

#Dataset
data=pd.read_csv('news_dataset.csv')
data.dropna(inplace=True)

print(data.shape)
print(data.head())

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

#Embeddings function
def sentence_embedding(sentence, data_dit):
  words = sentence.lower().split()
  word_embeddings = []
  for word in words:
    if word in data_dit:
      word_embeddings.append(data_dit[word])
  if not word_embeddings:
    return None

  sentence_embed = np.mean(word_embeddings, axis=0)
  return sentence_embed


data['Text'] = data['Text'].apply(wordopt)

news = data['Text']
y = data['label']

#Convert news to list
news=np.array(news)
news=news.tolist()

#Convert the whole dataset to embeddings
x=[]
for n in news:
  vector=sentence_embedding(n,data_dict)
  if vector is not None:
        x.append(vector.tolist())
  else:
    x.append([0.0] * 100)


xv_train,xv_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)

#LOGISTIC REGRESSION
LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr = LR.predict(xv_test)
lr_accuracy=accuracy_score(y_test,pred_lr)

print(classification_report(y_test, pred_lr))
print(f"Accuracy for LR model: {lr_accuracy}")


#DECISION TREES
DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt = DT.predict(xv_test)
dt_accuracy=accuracy_score(y_test,pred_dt)

print(f"Accuracy for DT model: {dt_accuracy}")


#GRADIENT BOOSTING
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train,y_train)
predict_gb = GB.predict(xv_test)
gb_accuracy=accuracy_score(y_test,predict_gb)

print(f"Accuracy for GB model: {gb_accuracy}")


#RANDOM FOREST
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train,y_train)
predict_rf = RF.predict(xv_test)
rf_accuracy=accuracy_score(y_test,predict_rf)

print(f"Accuracy for RF model: {rf_accuracy}")


#Save logistic regression model
dump(LR,"classifier_mod.pkl")

# Save as pickle file
with open(r"Fake-News-Detection\glove_100d.pkl", "wb") as f:
    pkl.dump(data_dict, f)