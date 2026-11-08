import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "life_expectancy_baseline.joblib")
FEATURES = list(MODEL.feature_names_in_)

DATA = pd.read_csv(ROOT / "Life Expectancy Data.csv")
DATA.columns = DATA.columns.str.strip()
DATA.columns = DATA.columns.str.strip()
SAMPLE = DATA.iloc[0].reindex(FEATURES).where(pd.notna(DATA.iloc[0].reindex(FEATURES)), None).to_dict()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Life Expectancy Prediction API"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["model_name"] == "LifeExpectancyRegressor"
    assert response.json()["model_loaded"] is True


def test_single_prediction(client):
    response = client.post("/predict", json={"data": SAMPLE})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert isinstance(response.json()["life_expectancy"], float)


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"records": [SAMPLE, SAMPLE]})
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_missing_features(client):
    response = client.post("/predict", json={"data": {"Country": "India"}})
    assert response.status_code == 422


def test_empty_batch(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422

