import gradio as gr
import requests

BACKEND = "http://backend:8200"

def ask(q: str):
    try:
        r = requests.post(f"{BACKEND}/ask", json={"question": q}, timeout=5)
        r.raise_for_status()
        return r.json().get("answer")
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Conversational Data Assistant (lite)")
    txt = gr.Textbox(label="Pregunta")
    btn = gr.Button("Preguntar")
    out = gr.Textbox(label="Respuesta")
    btn.click(ask, inputs=txt, outputs=out)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7862)
