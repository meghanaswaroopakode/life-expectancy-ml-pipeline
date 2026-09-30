from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET = "Life expectancy"
DATA_CANDIDATES = (
    ROOT_DIR / "data" / "Life Expectancy Data.csv",
    ROOT_DIR / "Life Expectancy Data.csv",
)
MODEL_PATH = ROOT_DIR / "models" / "life_expectancy_baseline.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "baseline_metrics.json"
MLFLOW_DB_PATH = ROOT_DIR / "artifacts" / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT = "life-expectancy-regression"
REGISTERED_MODEL_NAME = "LifeExpectancyRegressor"


def ensure_directories() -> None:
    for directory in ("models", "outputs", "logs", "artifacts", "mlruns"):
        (ROOT_DIR / directory).mkdir(parents=True, exist_ok=True)


def get_data_path() -> Path:
    for candidate in DATA_CANDIDATES:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(path.relative_to(ROOT_DIR)) for path in DATA_CANDIDATES)
    raise FileNotFoundError(f"Dataset not found. Expected it at {expected}.")


def load_data() -> pd.DataFrame:
    data = pd.read_csv(get_data_path())
    data.columns = [" ".join(str(column).split()) for column in data.columns]
    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    data = data.dropna(subset=[TARGET]).copy()
    integer_columns = data.select_dtypes(include=["integer"]).columns
    data[integer_columns] = data[integer_columns].astype("float64")
    return data


def split_data(data: pd.DataFrame | None = None):
    frame = load_data() if data is None else data.copy()
    features = frame.drop(columns=[TARGET])
    target = frame[TARGET]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numerical_columns = features.select_dtypes(include=[np.number]).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_model_pipeline(features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def regression_metrics(actual, predicted) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }