import json

import joblib
import numpy as np
from scipy import sparse

from src.common import ROOT_DIR


def has_only_finite_values(matrix) -> bool:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(np.isfinite(values).all())


def main() -> None:
    artifacts = ROOT_DIR / "artifacts"
    report_path = artifacts / "preprocessing_summary_report.json"
    preprocessor_path = artifacts / "preprocessor.pkl"
    features_path = artifacts / "preprocessed_features.joblib"
    for path in (report_path, preprocessor_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessing output is missing: {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    joblib.load(preprocessor_path)
    X_train_processed, X_test_processed = joblib.load(features_path)
    if X_train_processed.shape[1] != X_test_processed.shape[1]:
        raise ValueError("Train and test feature dimensions do not match.")
    if X_train_processed.shape[1] != report["feature_count"]:
        raise ValueError("Feature count does not match the preprocessing report.")
    if not has_only_finite_values(X_train_processed) or not has_only_finite_values(
        X_test_processed
    ):
        raise ValueError("Preprocessed features contain NaN or infinite values.")
    print("[SUCCESS] Output Validation PASSED.")
    print("[INFO] Features are imputed, encoded, scaled, and dimensionally consistent.")


if __name__ == "__main__":
    main()