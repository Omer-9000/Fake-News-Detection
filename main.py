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
from joblib import dump


#Dataset
data=pd.read_csv(r'Fake-News-Detection\news_dataset.csv')
data.dropna(inplace=True)

print(data.shape)
print(data.head())


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


data['Text'] = data['Text'].apply(wordopt)

x = data['Text']
y = data['label']

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=42)


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

print(xv_train.shape)

dump(vectorization,"vectorizer.pkl")

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr = LR.predict(xv_test)
lr_accuracy=accuracy_score(y_test,pred_lr)

print(classification_report(y_test, pred_lr))
print(f"Accuracy for LR model: {lr_accuracy}")

dump(LR,"linear_model.pkl")

DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt = DT.predict(xv_test)
dt_accuracy=accuracy_score(y_test,pred_dt)

print(f"Accuracy for DT model: {dt_accuracy}")

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train,y_train)
predict_gb = GB.predict(xv_test)
gb_accuracy=accuracy_score(y_test,predict_gb)

print(f"Accuracy for GB model: {gb_accuracy}")


RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train,y_train)
predict_rf = RF.predict(xv_test)
rf_accuracy=accuracy_score(y_test,predict_rf)

print(f"Accuracy for RF model: {rf_accuracy}")