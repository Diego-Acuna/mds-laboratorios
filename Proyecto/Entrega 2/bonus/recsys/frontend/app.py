import gradio as gr
import requests

BACKEND = "http://backend:8100"

def get_recs(customer_id: int):
    try:
        r = requests.get(f"{BACKEND}/recommend/{customer_id}?top_k=5", timeout=5)
        r.raise_for_status()
        j = r.json()
        recs = j.get("recommendations", [])
        return "\n".join([str(x) for x in recs])
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Recsys Demo")
    cid = gr.Number(label="Customer ID", value=25734)
    btn = gr.Button("Get recommendations")
    out = gr.Textbox(label="Recommendations")
    btn.click(get_recs, inputs=cid, outputs=out)

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7861)
