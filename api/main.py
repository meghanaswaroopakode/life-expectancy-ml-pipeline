from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "life_expectancy_baseline.joblib"

model = None
FEATURES = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, FEATURES
    model = joblib.load(MODEL_PATH)
    FEATURES = list(model.feature_names_in_)
    yield

app = FastAPI(
    title="Life Expectancy Prediction API",
    version="1.0.0",
    description="MLOps model serving API",
    lifespan=lifespan
)

class PredictionInput(BaseModel):
    data: dict

class BatchInput(BaseModel):
    records: list[dict]

def prepare(records):
    if not records:
        raise HTTPException(status_code=422, detail="Input records cannot be empty")

    df = pd.DataFrame(records)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})

    df = df[FEATURES]
    return df

@app.get("/")
def root():
    return {"service": "Life Expectancy Prediction API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy" if model is not None else "unavailable",
            "model_loaded": model is not None}

@app.get("/metadata")
def metadata():
    return {
        "model_name": "LifeExpectancyRegressor",
        "model_type": type(model.named_steps["model"]).__name__,
        "features": FEATURES,
        "target": "Life expectancy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(request: PredictionInput):
    try:
        df = prepare([request.data])
        result = float(model.predict(df)[0])
        return {"status": "success", "life_expectancy": round(result, 2), "unit": "years"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(request: BatchInput):
    try:
        df = prepare(request.records)
        predictions = model.predict(df).tolist()
        return {
            "status": "success",
            "count": len(predictions),
            "predictions": [round(float(x), 2) for x in predictions],
            "unit": "years"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
