import json

import joblib

from src.common import MODEL_PATH, ROOT_DIR, ensure_directories, regression_metrics, split_data


def main() -> None:
    ensure_directories()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Run the Lab 3 training step before evaluation.")
    _, X_test, _, y_test = split_data()
    model = joblib.load(MODEL_PATH)
    metrics = regression_metrics(y_test, model.predict(X_test))
    report_path = ROOT_DIR / "artifacts" / "evaluation_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("--- Regression Evaluation Report ---")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2  : {metrics['r2']:.4f}")
    print(f"[SUCCESS] Report saved to {report_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()