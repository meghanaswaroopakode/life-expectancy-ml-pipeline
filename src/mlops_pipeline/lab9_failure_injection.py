
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import sparse

from src.common import (
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    REGISTERED_MODEL_NAME,
    ROOT_DIR,
    TARGET,
    get_data_path,
)

REPETITIONS = 3
REPORT_PATH = ROOT_DIR / "reports" / "failure_report.json"


def run_module(module):
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode, output[-1500:]


def add_result(results, scenario, repetition, expected, actual, details):
    passed = expected == actual

    results.append({
        "scenario": scenario,
        "repetition": repetition,
        "expected": expected,
        "actual": actual,
        "status": "PASSED" if passed else "FAILED",
        "details": details,
    })

    print(
        f"[{'PASSED' if passed else 'FAILED'}] "
        f"{scenario} | Run {repetition} | {actual}"
    )


def backup_path(path):
    return path.with_name(path.name + ".lab9_backup")


def restore_file(original, backup):
    if backup.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists():
            original.unlink()
        shutil.move(str(backup), str(original))


def scenario_missing_dataset(results):
    print("\n[SCENARIO 1] Missing Dataset")

    dataset = get_data_path()
    backup = backup_path(dataset)

    for repetition in range(1, REPETITIONS + 1):
        try:
            shutil.move(str(dataset), str(backup))

            code, output = run_module("src.validate_data")
            actual = "HALT" if code != 0 else "PASSED"

            add_result(
                results, "Missing Dataset", repetition,
                "HALT", actual, output
            )
        finally:
            restore_file(dataset, backup)


def scenario_invalid_datatype(results):
    print("\n[SCENARIO 2] Invalid Numeric Datatype")

    dataset = get_data_path()
    backup = backup_path(dataset)

    for repetition in range(1, REPETITIONS + 1):
        shutil.copy2(dataset, backup)

        try:
            frame = pd.read_csv(dataset)
            frame.columns = [
                " ".join(str(column).split())
                for column in frame.columns
            ]

            numeric_columns = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if column != TARGET
            ]

            if not numeric_columns:
                raise ValueError("No numeric feature available.")

            column = numeric_columns[0]
            frame[column] = frame[column].astype(object)
            frame.loc[frame.index[0], column] = "INVALID_TYPE"
            frame.to_csv(dataset, index=False)

            # Validate through the actual preprocessing stage.
            code, output = run_module("src.validate_data")
            actual = "HALT" if code != 0 else "PASSED"

            add_result(
                results, "Invalid Numeric Datatype", repetition,
                "HALT", actual,
                f"Column: {column}\n{output}"
            )

        finally:
            restore_file(dataset, backup)


def scenario_schema_mismatch(results):
    print("\n[SCENARIO 3] Missing Target Column")

    dataset = get_data_path()
    backup = backup_path(dataset)

    for repetition in range(1, REPETITIONS + 1):
        shutil.copy2(dataset, backup)

        try:
            frame = pd.read_csv(dataset)

            frame.columns = [
                " ".join(str(column).split())
                for column in frame.columns
            ]

            frame.drop(columns=[TARGET], inplace=True)
            frame.to_csv(dataset, index=False)

            code, output = run_module("src.validate_data")
            actual = "HALT" if code != 0 else "PASSED"

            add_result(
                results, "Missing Target Column", repetition,
                "HALT", actual, output
            )

        finally:
            restore_file(dataset, backup)


def scenario_preprocessing_failure(results):
    print("\n[SCENARIO 4] NaN in Preprocessed Features")

    features_path = (
        ROOT_DIR / "artifacts" / "preprocessed_features.joblib"
    )
    backup = backup_path(features_path)

    if not features_path.is_file():
        raise FileNotFoundError(
            "Run the Lab 7 preprocessing pipeline first."
        )

    for repetition in range(1, REPETITIONS + 1):
        shutil.copy2(features_path, backup)

        try:
            X_train, X_test = joblib.load(features_path)

            if sparse.issparse(X_train):
                if X_train.data.size == 0:
                    raise ValueError("Training matrix is empty.")
                X_train.data[0] = np.nan
            else:
                X_train = np.asarray(X_train).copy()
                X_train[0, 0] = np.nan

            joblib.dump((X_train, X_test), features_path)

            code, output = run_module("src.validate_outputs")

            actual = "HALT" if code != 0 else "PASSED"

            add_result(
                results, "NaN in Preprocessed Features", repetition,
                "HALT", actual, output
            )

        finally:
            restore_file(features_path, backup)


def scenario_model_recovery(results):
    print("\n[SCENARIO 5] Missing Model and MLflow Recovery")

    backup = backup_path(MODEL_PATH)
    original_model_exists = MODEL_PATH.is_file()

    if backup.exists():
        backup.unlink()

    if original_model_exists:
        shutil.move(str(MODEL_PATH), str(backup))

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        for repetition in range(1, REPETITIONS + 1):

            code, output = run_module("src.evaluate")
            actual = "HALT" if code != 0 else "PASSED"

            add_result(
                results, "Missing Model Detection", repetition,
                "HALT", actual, output
            )

            recovered_model = mlflow.sklearn.load_model(
                f"models:/{REGISTERED_MODEL_NAME}@champion"
            )

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(recovered_model, MODEL_PATH)

            code, output = run_module("src.evaluate")
            actual = "RECOVERED" if code == 0 else "FAILED"

            add_result(
                results, "MLflow Champion Model Recovery", repetition,
                "RECOVERED", actual, output
            )

            if repetition < REPETITIONS:
                MODEL_PATH.unlink(missing_ok=True)

    finally:
        if backup.exists():
            restore_file(MODEL_PATH, backup)


def main():
    (ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 65)
    print("LAB 9 - FAILURE INJECTION AND RECOVERY")
    print("LIFE EXPECTANCY MLOPS")
    print("=" * 65)

    scenario_missing_dataset(results)
    scenario_invalid_datatype(results)
    scenario_schema_mismatch(results)
    scenario_preprocessing_failure(results)
    scenario_model_recovery(results)

    passed = sum(
        item["status"] == "PASSED" for item in results
    )
    failed = len(results) - passed

    report = {
        "lab": 9,
        "project": "Life Expectancy Prediction",
        "title": "Failure Injection and Recovery",
        "repetitions": REPETITIONS,
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": "PASSED" if failed == 0 else "FAILED",
        "results": results,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"Total Tests : {len(results)}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"LAB 9 STATUS: {report['overall_status']}")
    print(f"Report      : {REPORT_PATH}")
    print("=" * 65)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
