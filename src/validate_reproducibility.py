import json

import numpy as np

from src.common import ROOT_DIR, build_model_pipeline, ensure_directories, regression_metrics, split_data


def main() -> None:
    ensure_directories()
    X_train, X_test, y_train, y_test = split_data()
    results = []
    for _ in range(2):
        model = build_model_pipeline(X_train)
        model.fit(X_train, y_train)
        results.append(regression_metrics(y_test, model.predict(X_test)))

    first_metrics = list(results[0].values())
    second_metrics = list(results[1].values())
    if not np.allclose(first_metrics, second_metrics, rtol=0, atol=1e-12):
        raise AssertionError(f"Model runs are not reproducible: {results}")
    report = {"reproducible": True, "runs": results}
    report_path = ROOT_DIR / "artifacts" / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Reproducibility verified. Report: {report_path.name}")
    print("Metrics:", json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()