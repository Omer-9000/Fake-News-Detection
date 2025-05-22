import streamlit as st
import numpy as np
import re
import string
from joblib import load
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity

# Load classifier
classifier = load("C:\\Users\\AK\\Documents\\Fake-News-Detection\\classifier_mod.pkl")

# Load GloVe vectors
with open("C:\\Users\\AK\\Documents\\Fake-News-Detection\\glove_100d.pkl", "rb") as f:
    glove = pkl.load(f)

# Load Geo Fact Check title vectors
with open("C:\\Users\\AK\\Documents\\Fake-News-Detection\\glove_title_vectors.pkl", "rb") as f:
    titles, links, title_vectors = pkl.load(f)

# Preprocessing function
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

# Sentence embedding
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

def output_label(n):
  if n == 0:
    return 'Fake News'
  elif n == 1:
    return 'True News'

# Fact-check search
def check_fact(query):
    # Compare with cosine similarity
    scores = cosine_similarity(query, title_vectors)[0]
    top_indices = np.argsort(scores)[::-1][:3]

    # Show results
    for idx in top_indices:
        print(f"{scores[idx]:.2f} | {titles[idx]} → {links[idx]}")

# Streamlit UI
st.title("Fake News Detector")
user_input = st.text_area("Enter a news headline or short article:", height=150)

if st.button("Check Now"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline or short article.")
    else:
        clean_text = wordopt(user_input)
        vec = sentence_embedding(clean_text, glove).reshape(1, -1)

        pred = classifier.predict(vec)[0]
        st.subheader("Prediction:")
        st.success(output_label(pred))

        st.subheader("For more information:")
        results = check_fact(vec)

        for title, link, score in results:
            st.markdown(f"- [{title}]({link})  \n  _Similarity: {score:.2f}_")
