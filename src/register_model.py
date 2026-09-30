import json

import mlflow

from src.common import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    ROOT_DIR,
    ensure_directories,
    split_data,
)


def main() -> None:
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_report_path = ROOT_DIR / "artifacts" / "mlflow_run.json"
    if not run_report_path.is_file():
        raise FileNotFoundError("Run Lab 4 tracking before registering a model.")
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    registered = mlflow.register_model(run_report["model_uri"], REGISTERED_MODEL_NAME)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        registered.version,
    )
    champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    _, X_test, _, _ = split_data()
    predictions = champion.predict(X_test.iloc[:3])
    report = {
        "model_name": REGISTERED_MODEL_NAME,
        "version": str(registered.version),
        "alias": "champion",
        "source_run_id": run_report["run_id"],
        "smoke_test_predictions": [float(value) for value in predictions],
    }
    report_path = ROOT_DIR / "artifacts" / "model_registry_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[SUCCESS] Model registered and assigned alias 'champion'.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()