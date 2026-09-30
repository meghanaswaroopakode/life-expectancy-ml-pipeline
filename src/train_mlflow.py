import json

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.common import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    ROOT_DIR,
    build_model_pipeline,
    ensure_directories,
    regression_metrics,
    split_data,
)


def main() -> None:
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    X_train, X_test, y_train, y_test = split_data()
    model = build_model_pipeline(X_train)

    with mlflow.start_run(run_name="RandomForestRegressor") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = regression_metrics(y_test, predictions)
        mlflow.log_params(
            {
                "model": "RandomForestRegressor",
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "random_state": 42,
                "test_size": 0.2,
            }
        )
        mlflow.log_metrics(metrics)
        signature_input = X_test.copy()
        integer_columns = signature_input.select_dtypes(include=["integer"]).columns
        signature_input[integer_columns] = signature_input[integer_columns].astype("float64")
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(signature_input, predictions),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        run_report = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_uri": logged_model.model_uri,
            "metrics": metrics,
        }

    report_path = ROOT_DIR / "artifacts" / "mlflow_run.json"
    report_path.write_text(json.dumps(run_report, indent=2), encoding="utf-8")
    print(f"[SUCCESS] MLflow run tracked: {run.info.run_id}")
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()