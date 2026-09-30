# Life Expectancy ML Pipeline

This project predicts the numeric `Life expectancy` target. It uses the existing CSV in the project root, or `data/Life Expectancy Data.csv` if the dataset is moved there.

## Project layout

```text
data/                 Optional dataset directory
src/                  Data validation, preprocessing, training, evaluation
pipelines/            Lab 3-6 runners
models/               Baseline model output
outputs/              Existing EDA charts
logs/                 Pipeline execution logs
artifacts/            Metrics, preprocessing output, MLflow database, reports
mlruns/               MLflow model artifacts
```

## Run everything

From the project root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python .\pipelines\run_all.py
```

Individual labs can be run with `python .\pipelines\run_lab3_baseline.py` through `run_lab6_registry.py`. Lab 6 requires Lab 4 to have completed first.

The scripts use a fixed random seed, impute missing feature values within the training pipeline, and exclude rows without a target. Reports are written under `artifacts/`; run logs are written under `logs/`.