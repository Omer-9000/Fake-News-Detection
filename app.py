import streamlit as st
import numpy as np
import re
import string
from joblib import load
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from prediction import prediction_func, check_fact, output_label


model = SentenceTransformer('minilm_model') 

# Load classifier
classifier = load("C:\\Users\\AK\\Documents\\Fake-News-Detection\\classifier_log.pkl")

# Load scaler
scale = load("C:\\Users\\AK\\Documents\\Fake-News-Detection\\scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="", layout="centered")

# CSS Cleanup and Styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(145deg, #1f2937, #111827);
        color: #f3f4f6;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #60a5fa, #3b82f6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 1rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #d1d5db;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        background-color:#f3f4f6;
        color: #111827;
        border: 1px solid #d1d5db;
        font-size: 1rem;
        padding: 0.75rem;
        border-radius: 10px;
    }

    .stButton>button {
        background: linear-gradient(to right, #3b82f6, #6366f1);
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(to right, #60a5fa, #6366f1);
    }

    a {
        color: #93c5fd;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle"> Using AI to separate fact from fiction</div>', unsafe_allow_html=True)

# Input
user_input = st.text_area(" Paste your news content here:", height=150)

# Button logic
if st.button("🚀 Check Now"):
    if user_input.strip() == "":
        st.warning(" Please enter a news headline or short article.")
    else:
        st.subheader(" Prediction:")
        prediction = output_label(prediction_func(user_input, classifier, model, scale))
        if prediction.lower() == "real":
            st.success(" This news appears to be **REAL**.")
        else:
            st.error(" This news appears to be **FAKE**.")

            st.subheader("🔗 For Furthur Information:")
            results = check_fact(user_input, model, scale)
            if results:
                with st.expander("Related fact-checking results"):
                    for score, title, link in results:
                        st.markdown(f"- [{title}]({link})  \n  _Similarity Score: {score:.2f}_")
            else:
                st.info("No related fact-checks found.")
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)