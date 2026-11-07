
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

STAGES = [
    ("Data Validation", "src.validate_data"),
    ("Preprocessing", "src.preprocess_pipeline"),
    ("Output Validation", "src.validate_outputs"),
    ("Baseline Model Training", "src.train"),
    ("MLflow Training", "src.train_mlflow"),
    ("Model Evaluation", "src.evaluate"),
    ("Model Registration", "src.register_model"),
    ("Reproducibility Validation", "src.validate_reproducibility"),
]


def main():
    (ROOT_DIR / "logs").mkdir(exist_ok=True)
    (ROOT_DIR / "reports").mkdir(exist_ok=True)

    results = []

    print("=" * 65)
    print("LAB 7 - LIFE EXPECTANCY END-TO-END ML PIPELINE")
    print("=" * 65)

    for stage, module in STAGES:
        print(f"\n[INFO] Running: {stage}")

        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT_DIR,
            text=True
        )

        status = "PASSED" if result.returncode == 0 else "FAILED"

        results.append({
            "stage": stage,
            "module": module,
            "status": status,
            "exit_code": result.returncode
        })

        print(f"[{status}] {stage}")

        if result.returncode != 0:
            break

    overall = (
        "PASSED"
        if len(results) == len(STAGES)
        and all(item["status"] == "PASSED" for item in results)
        else "FAILED"
    )

    report = {
        "lab": 7,
        "project": "Life Expectancy Prediction",
        "pipeline": "End-to-End ML Pipeline",
        "stages": results,
        "overall_status": overall
    }

    report_path = ROOT_DIR / "reports" / "lab7_pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"LAB 7 STATUS: {overall}")
    print(f"Report saved: {report_path}")
    print("=" * 65)

    sys.exit(0 if overall == "PASSED" else 1)


if __name__ == "__main__":
    main()
