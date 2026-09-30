from _runner import run_steps


if __name__ == "__main__":
    run_steps(
        "Lab 4: MLflow Experiment Tracking",
        ["src.train_mlflow", "src.validate_reproducibility"],
    )