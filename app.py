import streamlit as st
import pandas as pd
import joblib
import spacy
from transformers import pipeline
import os
import subprocess

# Function to ensure spaCy model is installed
def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"Downloading {model_name} model, this may take some time...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

# Load models
ner_model = load_spacy_model()  # Named Entity Recognition
sentiment_analyzer = pipeline("sentiment-analysis")  # Sentiment Analysis

# Load text classification model
try:
    classifier = joblib.load("text_classifier.pkl")
except Exception as e:
    st.error("Failed to load text classifier model. Ensure `text_classifier.pkl` is in the project directory.")
    st.stop()

# Streamlit UI
st.title("AI-Powered Text Processing App")
st.write("Upload documents for OCR, classification, and analysis.")

# File Upload
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Extracted Text:")
    st.text_area("File Content", text, height=200)

    # Named Entity Recognition (NER)
    doc = ner_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    st.subheader("Named Entities:")
    st.write(pd.DataFrame(entities, columns=["Entity", "Category"]))

    # Sentiment Analysis
    sentiment = sentiment_analyzer(text[:512])  # Limiting to 512 chars
    st.subheader("Sentiment Analysis:")
    st.write(sentiment)

    # Text Classification
    prediction = classifier.predict([text])[0]
    st.subheader("Text Classification:")
    st.write(f"Predicted Category: **{prediction}**")

# Run using: streamlit run app.py
