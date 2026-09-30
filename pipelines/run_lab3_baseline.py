from _runner import run_steps


if __name__ == "__main__":
    run_steps(
        "Lab 3: Baseline ML Pipeline",
        ["src.validate_data", "src.train", "src.evaluate"],
    )