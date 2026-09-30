import json

import joblib

from src.common import (
    METRICS_PATH,
    MODEL_PATH,
    ROOT_DIR,
    build_model_pipeline,
    ensure_directories,
    regression_metrics,
    split_data,
)


def main() -> None:
    ensure_directories()
    X_train, X_test, y_train, y_test = split_data()
    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Baseline model saved to {MODEL_PATH.relative_to(ROOT_DIR)}")
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()