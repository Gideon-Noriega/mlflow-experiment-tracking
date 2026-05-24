"""FastAPI model serving gateway."""
import logging, os, time
from contextlib import asynccontextmanager
import mlflow, pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.serving.ab_router import ABRouter

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    tenure_months: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_support_tickets: int = Field(..., ge=0)
    days_since_last_login: int = Field(..., ge=0)
    num_logins_last_30d: int = Field(..., ge=0)
    contract_value: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    contract_type: str
    payment_method: str
    subscription_tier: str
    onboarding_completed: str


class PredictionResponse(BaseModel):
    prediction: int; churn_probability: float
    model_version: str; model_alias: str; latency_ms: float


ab_router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ab_router
    tu = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mn = os.getenv("MODEL_NAME", "customer-churn-model")
    mlflow.set_tracking_uri(tu)
    ab_router = ABRouter(mn, tu, float(os.getenv("AB_CHASPION_WEIGHT", "0.9")),
                          float(os.getenv("AB_CHALLENGER_WEIGHT", "0.1")))
    ab_router.load_models()
    yield


app = FastAPI(title="Customer Churn API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if not ab_router or not ab_router.is_ready(): raise HTTPException(503)
    return {"status": "healthy"}


@app.get("/model-info")
async def model_info():
    return {"name": ab_router.model_name, "champion": ab_router.champion_version,
            "challenger": ab_router.challenger_version}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    if not ab_router: raise HTTPException(503)
    start = time.time()
    df = pd.DataFrame([req.model_dump()])
    model, alias, ver = ab_router.route()
    try:
        p = model.predict(df)
        pv = int(p[0]) if p.ndim == 1 else int(p[0][0])
    except Exception as e: raise HTTPException(500, str(e))
    lat = (time.time()-start)*1000
    ab_router.log_prediction(alias, pv, lat)
    return PredictionResponse(prediction=pv, churn_probability=0.5,
        model_version=ver, model_alias=alias, latency_ms=round(lat, 2))


@app.post("/reload")
async def reload():
    if ab_router: ab_router.load_models()
    return {"status": "reloaded"}
