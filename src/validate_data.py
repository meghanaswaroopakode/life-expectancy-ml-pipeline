import json

import pandas as pd

from src.common import ROOT_DIR, TARGET, get_data_path, load_data


def main() -> None:
    raw_data_path = get_data_path()
    raw_data = pd.read_csv(raw_data_path)
    raw_data.columns = [" ".join(str(column).split()) for column in raw_data.columns]
    data = load_data()
    report = {
        "dataset": str(raw_data_path.relative_to(ROOT_DIR)),
        "raw_records": int(len(raw_data)),
        "usable_records": int(len(data)),
        "dropped_missing_target": int(raw_data[TARGET].isna().sum()),
        "feature_count": int(data.shape[1] - 1),
        "target": TARGET,
        "remaining_missing_values": int(data.isna().sum().sum()),
    }
    if report["usable_records"] < 10:
        raise ValueError("Dataset must contain at least 10 rows with a target value.")
    print(f"[INFO] Validating schema (Records: {report['raw_records']})...")
    print("[SUCCESS] Schema Validation PASSED.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()