import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_steps(title: str, modules: list[str]) -> None:
    (ROOT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_path = ROOT_DIR / "logs" / f"{Path(sys.argv[0]).stem}.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT_DIR) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("GIT_PYTHON_REFRESH", "quiet")
    print(f"[INFO] Starting {title}")
    for module in modules:
        print(f"\n[INFO] ---> Executing {module}...")
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT_DIR,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- {module} (exit {result.returncode}) ---\n")
            log_file.write(result.stdout)
        if result.returncode:
            raise SystemExit(f"[ERROR] {module} failed with exit code {result.returncode}.")
    print(f"\n[SUCCESS] {title} fully executed!")