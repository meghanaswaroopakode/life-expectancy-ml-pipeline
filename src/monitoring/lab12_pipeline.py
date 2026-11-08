import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from src.common import (
    ROOT_DIR, MODEL_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    load_data, split_data, build_model_pipeline, regression_metrics,
    ensure_directories,
)

def save_report(name, data):
    path = ROOT_DIR / "reports" / "lab12" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[SAVED] {path}")

def main():
    print("=" * 65)
    print("LAB 12 - MODEL MONITORING, DRIFT DETECTION AND RETRAINING")
    print("=" * 65)

    ensure_directories()
    X_train, X_test, y_train, y_test = split_data()

    # 1. Load deployed baseline model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

    baseline = joblib.load(MODEL_PATH)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = regression_metrics(y_test, baseline_predictions)

    save_report("performance_monitoring.json", {
        "model": "LifeExpectancyRegressor",
        "metrics": baseline_metrics,
        "test_records": len(X_test),
        "status": "MONITORED",
    })

    # 2. Data drift detection: compare training and test feature distributions
    drift_results = {}
    for column in X_train.select_dtypes(include=np.number).columns:
        train_mean = float(X_train[column].mean())
        test_mean = float(X_test[column].mean())
        train_std = float(X_train[column].std())
        difference = abs(test_mean - train_mean)
        drift_score = difference / train_std if train_std > 0 else 0.0

        drift_results[column] = {
            "training_mean": train_mean,
            "test_mean": test_mean,
            "standardized_mean_difference": float(drift_score),
            "drift_detected": bool(drift_score > 0.5),
        }

    save_report("data_drift.json", {
        "comparison": "training_data_vs_test_data",
        "features": drift_results,
        "drifted_features": [
            name for name, result in drift_results.items()
            if result["drift_detected"]
        ],
    })

    # 3. Prediction drift monitoring
    train_predictions = baseline.predict(X_train)
    prediction_report = {
        "training_prediction_mean": float(np.mean(train_predictions)),
        "test_prediction_mean": float(np.mean(baseline_predictions)),
        "training_prediction_std": float(np.std(train_predictions)),
        "test_prediction_std": float(np.std(baseline_predictions)),
        "prediction_mean_difference": float(
            abs(np.mean(train_predictions) - np.mean(baseline_predictions))
        ),
    }
    save_report("prediction_drift.json", prediction_report)

    # 4. Target distribution comparison (concept-drift proxy)
    target_report = {
        "training_target_mean": float(y_train.mean()),
        "test_target_mean": float(y_test.mean()),
        "training_target_std": float(y_train.std()),
        "test_target_std": float(y_test.std()),
        "mean_difference": float(abs(y_train.mean() - y_test.mean())),
        "note": "Target-distribution comparison only; not proof of temporal concept drift.",
    }
    save_report("target_drift_analysis.json", target_report)

    # 5. Train and compare a candidate model
    candidate = build_model_pipeline(X_train)
    candidate.fit(X_train, y_train)
    candidate_predictions = candidate.predict(X_test)
    candidate_metrics = regression_metrics(y_test, candidate_predictions)

    comparison = {
        "baseline_model": baseline_metrics,
        "candidate_model": candidate_metrics,
        "candidate_improved_mae": bool(
            candidate_metrics["mae"] < baseline_metrics["mae"]
        ),
        "candidate_improved_rmse": bool(
            candidate_metrics["rmse"] < baseline_metrics["rmse"]
        ),
    }
    save_report("model_comparison.json", comparison)

    # 6. Log candidate retraining run in MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="Lab12_Monitoring_Retraining") as run:
        mlflow.log_param("lab", 12)
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("training_records", len(X_train))
        mlflow.log_param("test_records", len(X_test))
        mlflow.log_metrics({
            f"candidate_{key.lower()}": value
            for key, value in candidate_metrics.items()
        })
        mlflow.sklearn.log_model(candidate, name="lab12_candidate", skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])

        retraining_report = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "candidate_metrics": candidate_metrics,
            "model_saved": True,
            "registry_updated": False,
            "note": "Candidate logged to MLflow; production model was not replaced.",
        }

    save_report("retraining_report.json", retraining_report)

    # 7. Final monitoring summary
    summary = {
        "lab": 12,
        "status": "PASSED",
        "checks": {
            "baseline_performance_monitoring": "PASSED",
            "feature_drift_analysis": "PASSED",
            "prediction_drift_monitoring": "PASSED",
            "target_distribution_analysis": "PASSED",
            "candidate_model_training": "PASSED",
            "mlflow_candidate_logging": "PASSED",
        },
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "reports_directory": "reports/lab12/",
    }
    save_report("lab12_summary.json", summary)

    print("\n[ SUCCESS ] LAB 12 PIPELINE COMPLETED")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()


