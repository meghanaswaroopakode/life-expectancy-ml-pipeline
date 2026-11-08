
import json
import pandas as pd

from src.common import ROOT_DIR, TARGET, get_data_path, load_data


NUMERIC_COLUMNS = [
    "Year",
    "Life expectancy",
    "Adult Mortality",
    "infant deaths",
    "Alcohol",
    "percentage expenditure",
    "Hepatitis B",
    "Measles",
    "BMI",
    "under-five deaths",
    "Polio",
    "Total expenditure",
    "Diphtheria",
    "HIV/AIDS",
    "GDP",
    "Population",
    "thinness 1-19 years",
    "thinness 5-9 years",
    "Income composition of resources",
    "Schooling",
]


def main() -> None:
    raw_data_path = get_data_path()

    raw_data = pd.read_csv(raw_data_path)

    # Normalize column names
    raw_data.columns = [
        " ".join(str(column).split())
        for column in raw_data.columns
    ]

    # Validate target column
    if TARGET not in raw_data.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing."
        )

    # Validate numeric columns
    for column in NUMERIC_COLUMNS:
        if column not in raw_data.columns:
            continue

        converted = pd.to_numeric(
            raw_data[column],
            errors="coerce"
        )

        invalid_values = (
            raw_data[column].notna() & converted.isna()
        )

        if invalid_values.any():
            raise ValueError(
                f"Invalid numeric datatype detected in column: {column}"
            )

    # Load cleaned dataset
    data = load_data()

    report = {
        "dataset": str(raw_data_path.relative_to(ROOT_DIR)),
        "raw_records": int(len(raw_data)),
        "usable_records": int(len(data)),
        "dropped_missing_target": int(
            raw_data[TARGET].isna().sum()
        ),
        "feature_count": int(data.shape[1] - 1),
        "target": TARGET,
        "remaining_missing_values": int(
            data.isna().sum().sum()
        ),
        "numeric_datatypes_validated": True,
    }

    if report["usable_records"] < 10:
        raise ValueError(
            "Dataset must contain at least 10 rows with a target value."
        )

    print(
        f"[INFO] Validating schema (Records: {report['raw_records']})..."
    )
    print("[SUCCESS] Schema Validation PASSED.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
