import streamlit as st
import pandas as pd
import joblib
import spacy
from transformers import pipeline

import os
import subprocess
import spacy

# Ensure spaCy model is installed before use
try:
    spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz"])
    spacy.load("en_core_web_sm")  # Try loading again after installation

# Now use the model
ner_model = spacy.load("en_core_web_sm")  # Named Entity Recognition



# Load NLP models
ner_model = spacy.load("en_core_web_sm")  # Named Entity Recognition
sentiment_analyzer = pipeline("sentiment-analysis")  # Sentiment Analysis
classifier = joblib.load("text_classifier.pkl")  # Pre-trained text classification model

st.title("AI-Powered Text Processing App")
st.write("Upload documents for OCR, classification, and analysis.")

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Read file content
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Extracted Text:")
    st.text_area("File Content", text, height=200)

    # Named Entity Recognition (NER)
    doc = ner_model(text)
    st.subheader("Named Entities:")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(pd.DataFrame(entities, columns=["Entity", "Category"]))

    # Sentiment Analysis
    sentiment = sentiment_analyzer(text[:512])  # Limiting to 512 chars for efficiency
    st.subheader("Sentiment Analysis:")
    st.write(sentiment)

    # Text Classification
    prediction = classifier.predict([text])[0]
    st.subheader("Text Classification:")
    st.write(f"Predicted Category: **{prediction}**")

# Run using: streamlit run app.py
