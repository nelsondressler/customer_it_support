import os
import requests

import gradio as gr

import config

# URL of the locally running API server
API_URL = config.API_URL

def predict_email(subject, body, model_choice):
    payload = {"subject": subject, "body": body, "model_choice": model_choice}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return f"Queue: {data['queue']}\nPriority: {data['priority']}\nDetails: {data['details']}"
    else:
        return f"Error: {response.text}"

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_email,
    inputs=[
        gr.Textbox(label="Subject"),
        gr.Textbox(label="Body", lines=4),
        gr.Radio(choices=["nb", "lr", "distilbert", "bert"], label="Model Choice")
    ],
    outputs="text",
    title="Customer IT Support Email Classifier"
)

if __name__ == "__main__":
    iface.launch()