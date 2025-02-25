import streamlit as st
import pandas as pd
import joblib
import spacy
from transformers import pipeline

import streamlit as st
import pandas as pd
import joblib
import spacy
from transformers import pipeline
import os  # Import the os module

# Download spaCy model if it's not already present
model_name = "en_core_web_sm"
model_path = os.path.join(os.getcwd(), model_name)  # Create a path in the current directory

if not os.path.exists(model_path):
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name])
    except Exception as e:
        st.error(f"Failed to download spaCy model: {e}")
        st.stop()

# Load the spaCy model from the specified path
try:
    ner_model = spacy.load(model_path)
except OSError:
    st.error(f"spaCy model '{model_name}' could not be loaded from '{model_path}'.")
    st.stop()

# Load NLP models
sentiment_analyzer = pipeline("sentiment-analysis")  # Sentiment Analysis

# Load text classification model
try:
    classifier = joblib.load("text_classifier.pkl")
except Exception as e:
    st.error("Failed to load text classifier model. Ensure 'text_classifier.pkl' is present.")
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
