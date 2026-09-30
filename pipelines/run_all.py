import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LABS = (
    "run_lab3_baseline.py",
    "run_lab4_tracking.py",
    "run_lab5_pipeline.py",
    "run_lab6_registry.py",
)


def main() -> None:
    (ROOT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_path = ROOT_DIR / "logs" / "run_all.log"
    for lab in LABS:
        print(f"\n[INFO] Running {lab}...")
        result = subprocess.run(
            [sys.executable, str(ROOT_DIR / "pipelines" / lab)],
            cwd=ROOT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- {lab} (exit {result.returncode}) ---\n")
            log_file.write(result.stdout)
        if result.returncode:
            raise SystemExit(f"[ERROR] {lab} failed with exit code {result.returncode}.")
    print("\n[SUCCESS] Labs 3-6 all completed.")


if __name__ == "__main__":
    main()