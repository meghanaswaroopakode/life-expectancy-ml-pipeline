import json

import joblib

from src.common import ROOT_DIR, build_preprocessor, ensure_directories, split_data


def main() -> None:
    ensure_directories()
    X_train, X_test, _, _ = split_data()
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    preprocessor_path = ROOT_DIR / "artifacts" / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(
        (X_train_processed, X_test_processed),
        ROOT_DIR / "artifacts" / "preprocessed_features.joblib",
    )
    report = {
        "train_rows": int(X_train_processed.shape[0]),
        "test_rows": int(X_test_processed.shape[0]),
        "feature_count": int(X_train_processed.shape[1]),
        "preprocessor": str(preprocessor_path.relative_to(ROOT_DIR)),
    }
    report_path = ROOT_DIR / "artifacts" / "preprocessing_summary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[SUCCESS] Production preprocessing completed.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()