import streamlit as st
from transformers import pipeline

st.title("📰 Misinformation Detection")

classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

text = st.text_area("Enter text to check:")

if st.button("Check"):
    if text:
        result = classifier(text)[0]
        st.success(f"{result['label']} (Confidence: {result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
