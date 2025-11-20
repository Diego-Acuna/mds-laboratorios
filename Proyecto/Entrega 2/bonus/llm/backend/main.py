from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re

APP = FastAPI(title="LLM-lite Backend")

if Path("/data").exists():
    DATA_DIR = Path("/data")
else:
    try:
        DATA_DIR = Path(__file__).resolve().parents[4] / "airflow" / "data"
    except Exception:
        DATA_DIR = Path.cwd() / "airflow" / "data"
TRANS_PATH = DATA_DIR / "transacciones.parquet"
CLIENTS_PATH = DATA_DIR / "clientes.parquet"
PRODUCTS_PATH = DATA_DIR / "productos.parquet"

_trans = None
_clients = None
_products = None


def _ensure():
    global _trans, _clients, _products
    if _trans is None:
        if not TRANS_PATH.exists():
            raise FileNotFoundError("transacciones not found")
        _trans = pd.read_parquet(TRANS_PATH)
    if _clients is None:
        _clients = pd.read_parquet(CLIENTS_PATH)
    if _products is None:
        _products = pd.read_parquet(PRODUCTS_PATH)


class AskRequest(BaseModel):
    question: str


@APP.get('/health')
def health():
    ok = TRANS_PATH.exists()
    return {"status": "ok", "data_found": ok}


@APP.post('/ask')
def ask(req: AskRequest):
    q = req.question.lower()
    try:
        _ensure()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # simple rule-based handlers
    if 'clientes únicos' in q or 'clientes unicos' in q or 'clientes únicos' in q:
        return {"answer": int(_clients['customer_id'].nunique())}

    m = re.search(r'transacciones ha realizado el cliente\s*(\d+)', q)
    if m:
        cid = int(m.group(1))
        n = int((_trans[_trans['customer_id'] == cid].shape[0]))
        return {"answer": f"Cliente {cid} realizó {n} transacciones"}

    if 'productos únicos' in q or 'productos unicos' in q:
        return {"answer": int(_products['product_id'].nunique())}

    # fallback: provide a short summary
    return {"answer": "No entiendo la pregunta. Preguntas de ejemplo: '¿Cuántos clientes únicos hay?', '¿Cuántas transacciones ha realizado el cliente 12345?', '¿Cuántos productos únicos?'"}
