import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import re
import string
from bs4 import BeautifulSoup
import requests
from difflib import SequenceMatcher
from joblib import dump
import pickle as pkl
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('Fake-News-Detection\minilm_model')


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


data['Text'] = data['Text'].apply(wordopt)

news = data['Text']
y = data['label']

#Convert news to list
news=np.array(news)
news=news.tolist()

#Convert the whole dataset to embeddings
embeddings = model.encode(news, convert_to_tensor=True)
embeddings=embeddings.tolist()

xv_train,xv_test,y_train,y_test = train_test_split(embeddings, y, test_size=0.2,random_state=42)

scaler = StandardScaler()
xv_train = scaler.fit_transform(xv_train)
xv_test = scaler.transform(xv_test)

#LOGISTIC REGRESSION
LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr = LR.predict(xv_test)
lr_accuracy=accuracy_score(y_test,pred_lr)
lr_precision=precision_score(y_test,pred_lr)
lr_recall=recall_score(y_test,pred_lr)
lr_f1=f1_score(y_test,pred_lr)

print(f"Accuracy for LR model: {lr_accuracy}")
print(f"Precision for LR model: {lr_precision}")
print(f"Recall for LR model: {lr_recall}")
print(f"F1-score for LR model: {lr_f1}")


#DECISION TREES
DT = DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt = DT.predict(xv_test)
dt_accuracy=accuracy_score(y_test,pred_dt)
dt_precision=precision_score(y_test,pred_dt)
dt_recall=recall_score(y_test,pred_dt)
dt_f1=f1_score(y_test,pred_dt)

print(f"Accuracy for DT model: {dt_accuracy}")
print(f"Precision for DT model: {dt_precision}")
print(f"Recall for DT model: {dt_recall}")
print(f"F1-score for DT model: {dt_f1}")


#RANDOM FOREST
RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train,y_train)
predict_rf = RF.predict(xv_test)
rf_accuracy=accuracy_score(y_test,predict_rf)
rf_precision=precision_score(y_test,predict_rf)
rf_recall=recall_score(y_test,predict_rf)
rf_f1=f1_score(y_test,predict_rf)

print(f"Accuracy for RF model: {rf_accuracy}")
print(f"Precision for RF model: {rf_precision}")
print(f"Recall for RF model: {rf_recall}")
print(f"F1-score for RF model: {rf_f1}")


#SVM
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(xv_train,y_train)
pred_svc = svc_model.predict(xv_test)
svc_accuracy=accuracy_score(y_test,pred_svc)
svc_precision=precision_score(y_test,pred_svc)
svc_recall=recall_score(y_test,pred_svc)
svc_f1=f1_score(y_test,pred_svc)

print(f"Accuracy for SVC model: {svc_accuracy}")
print(f"Precision for SVC model: {svc_precision}")
print(f"Recall for SVC model: {svc_recall}")
print(f"F1-score for SVC model: {svc_f1}")


#Save logistic regression model
dump(LR,"classifier_log.pkl")

#Save scaler
dump(scaler,"scaler.pkl")

