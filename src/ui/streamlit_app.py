import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Customer IT Support Email Classifier")  # Matching Gradio's title

subject = st.text_input("Subject")  # Simplified label
body = st.text_area("Body", height=150)  # Adjust height as needed
model_choice = st.radio("Model Choice", options=["nb", "lr", "distilbert", "bert"])  # Using radio buttons

if st.button("Classify"):
    payload = {"subject": subject, "body": body, "model_choice": model_choice}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        st.write(f"Queue: {data['queue']}\nPriority: {data['priority']}\nDetails: {data['details']}")  # Displaying all details
    else:
        st.error(f"Error: {response.text}")  # Displaying error details