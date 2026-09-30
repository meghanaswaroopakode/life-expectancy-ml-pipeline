from _runner import run_steps


if __name__ == "__main__":
    run_steps(
        "Lab 5: Production Data Pipeline",
        ["src.validate_data", "src.preprocess_pipeline", "src.validate_outputs"],
    )