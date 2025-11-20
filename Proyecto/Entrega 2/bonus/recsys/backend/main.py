from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

APP = FastAPI(title="Recsys Backend")

# Data location (re-use airflow data)
# Prefer the docker-mounted path `/data` if present, otherwise fall back
# to repo-relative locations or the current working dir.
if Path("/data").exists():
    DATA_DIR = Path("/data")
else:
    try:
        DATA_DIR = Path(__file__).resolve().parents[4] / "airflow" / "data"
    except Exception:
        DATA_DIR = Path.cwd() / "airflow" / "data"

TRANS_PATH = DATA_DIR / "transacciones.parquet"

# simple in-memory caches
_df = None
_cust_top = None


def _ensure_data():
    global _df, _cust_top
    if _df is None:
        if not TRANS_PATH.exists():
            raise FileNotFoundError(f"transactions not found at {TRANS_PATH}")
        _df = pd.read_parquet(TRANS_PATH)
        # compute top products per customer
        grp = _df.groupby(["customer_id", "product_id"]).size().reset_index(name="cnt")
        cust_groups = grp.sort_values(["customer_id", "cnt"], ascending=[True, False])
        _cust_top = (
            cust_groups.groupby("customer_id").apply(lambda d: d.head(20)["product_id"].tolist()).to_dict()
        )


class RecommendResponse(BaseModel):
    customer_id: int
    recommendations: list


@APP.get("/health")
def health():
    ok = TRANS_PATH.exists()
    return {"status": "ok", "data_found": ok}


@APP.get("/recommend/{customer_id}")
def recommend(customer_id: int, top_k: int = 5):
    try:
        _ensure_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # If customer seen before, return their top products; otherwise return global top
    products = _cust_top.get(customer_id)
    if products is None:
        # global popularity
        global_top = (
            _df["product_id"].value_counts().nlargest(top_k).index.tolist()
        )
        return RecommendResponse(customer_id=customer_id, recommendations=global_top[:top_k]).dict()
    else:
        return RecommendResponse(customer_id=customer_id, recommendations=products[:top_k]).dict()
