import os
import json
import requests
import gradio as gr
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

def predict_from_text(text: str):
    # Expecting JSON array of dicts, e.g. [{"f1": 1, "f2": 2}]
    try:
        payload = json.loads(text)
    except Exception as e:
        return f"Failed to parse JSON: {e}"
    if not isinstance(payload, list):
        return "Please provide a JSON array of records (list of objects)."
    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json={"instances": payload}, timeout=15)
        if resp.status_code != 200:
            return f"Backend error {resp.status_code}: {resp.text}"
        data = resp.json()
        return data
    except Exception as e:
        return f"Request failed: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Sodai Predictions (Gradio)")
    txt = gr.Textbox(lines=10, placeholder='Paste JSON array of records here')
    out = gr.Textbox(lines=6)
    btn = gr.Button("Predict")
    btn.click(fn=predict_from_text, inputs=txt, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
