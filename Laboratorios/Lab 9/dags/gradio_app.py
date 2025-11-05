import os
import time
import gradio as gr
from hiring_functions import _latest_model_under_runs, predict

def main():
    port = int(os.getenv("GRADIO_PORT", "7860"))
    keep = int(os.getenv("GRADIO_KEEP_SECONDS", "1200"))

    model_path = _latest_model_under_runs()
    if not model_path:
        raise FileNotFoundError("No hay modelo en runs/*/models/. Ejecuta preprocess_and_train primero.")

    ui = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada."
    )

    try:
        # 1) intenta link público; si la red bloquea frpc, cae a local
        ui.launch(server_name="0.0.0.0", server_port=port,
                  share=True, inbrowser=False, prevent_thread_lock=True)
        print("[gradio] Lanzado con share=True. Revisa el enlace *.gradio.live en el log.")
    except Exception as e:
        print(f"[gradio] No se pudo crear link público ({e}). Usando URL local: http://localhost:{port}")
        ui.launch(server_name="0.0.0.0", server_port=port,
                  share=False, inbrowser=False, prevent_thread_lock=True)
        print(f"[gradio] URL local: http://localhost:{port}")

    # Mantener la app viva 'keep' segundos
    for rem in range(keep, 0, -10):
        if rem % 60 == 0:
            print(f"[gradio] Ventana activa ~{rem//60} min restantes. http://localhost:{port}")
        time.sleep(10)

    print("[gradio] Ventana finalizada. Cerrando servidor.")

if __name__ == "__main__":
    main()