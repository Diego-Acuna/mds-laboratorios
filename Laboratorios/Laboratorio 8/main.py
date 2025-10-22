# Importamos librerías necesarias
from typing import Optional
import os
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# Obtenemos la ruta donde optimize.py guardó el modelo
MODEL_PATH = "models/model.pkl"

# Definimos la app de FastAPI
app = FastAPI(title="Potabilidad API", description="API para predecir potabilidad del agua con XGBoost optimizado", version="1.0")

# Definimos el esquema de entrada esperado
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float = Field(..., alias="Organic_carbon")
    Trihalomethanes: float
    Turbidity: float

# Cargamos el modelo en memoria al iniciar la app
model = None
FEATURE_ORDER = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]

def load_model(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Model file not found at {path}. Asegúrate de ejecutar optimize_model primero.")

# Evento de inicio para cargar el modelo
@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"Modelo cargado desde {MODEL_PATH}")
    except Exception as e:
        # Dejar app corriendo pero las predicciones lanzarán HTTP 500
        model = None
        print(f"Advertencia: no se pudo cargar el modelo: {e}")

# Definimos los endpoints
# Endpoint raíz con información del servicio
@app.get("/")
def home():
    return {
        "service": "Potability prediction",
        "description": "Recibe mediciones químicas del agua y devuelve 0/1 (no potable / potable).",
        "input_example": {
            "ph": 7.0,
            "Hardness": 150.0,
            "Solids": 10000.0,
            "Chloramines": 3.0,
            "Sulfate": 300.0,
            "Conductivity": 400.0,
            "Organic_carbon": 10.0,
            "Trihalomethanes": 70.0,
            "Turbidity": 3.0
        },
        "note": "Usa POST /potabilidad/ para predecir. Visita /docs para UI."
    }

# Endpoint de predicción de potabilidad
@app.post("/potabilidad/")
def predict(sample: WaterSample):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible. Ejecuta optimize_model y asegúrate de tener models/model.pkl")
    # Convertir a DataFrame en el orden esperado
    try:
        row = {k: getattr(sample, k) for k in FEATURE_ORDER}
    except Exception:
        # pydantic asegura keys, pero seguridad extra
        raise HTTPException(status_code=400, detail="JSON inválido o faltan campos requeridos.")
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    try:
        pred = model.predict(X)
        potabilidad = int(pred[0])
        return {"potabilidad": potabilidad}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

if __name__ == "__main__":
    import uvicorn
    # Ejecuta el servidor al correr "python main.py"
    uvicorn.run(app, host="0.0.0.0", port=8000)