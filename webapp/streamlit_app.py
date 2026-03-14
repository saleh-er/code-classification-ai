import streamlit as st
from src.predict import predict

st.title("AI Code Language Detector")

code = st.text_area("Paste your code")

if st.button("Predict Language"):
    result = predict(code)
    st.success(f"Detected language: {result}")
