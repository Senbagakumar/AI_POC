# Part 1 — Data Audit, EDA & Business Understanding

This folder contains a reproducible implementation of the raw-data audit and exploratory analysis requested in Part 1 of the capstone.

## Deliverables

- `eda_audit.ipynb`
- `data_quality_report.md`
- `business_memo.md`
- `charts/` with the analysis visuals used in the notebook and reports
- `build_part1.py` to regenerate all outputs
- `requirements.txt`

## Data Loading

The script looks for the dataset in either of these relative locations:

1. `PART_1/data/`
2. `../d2c churn data package/d2c churn data package/`

If you move this folder into its own repository, place the seven churn CSV files inside `data/`.

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_part1.py
```

## What The Script Produces

- A data-quality report covering package inspection, missing values, duplicate-like rows, invalid values, outliers, join/key issues, date consistency checks, and leakage-sensitive columns.
- A business memo with concrete pre-campaign investigations.
- A notebook that includes package inspection, exploratory tables, the generated charts, churn-risk hypotheses, and the business memo.

## Integrity Notes

- All customer-level analysis uses real customer IDs and observed churn labels from the provided dataset.
- The script treats `2025-09-30` as the snapshot boundary and only uses post-snapshot orders to flag leakage risk, not as usable features.
- All paths are relative. No credentials, tokens, or machine-specific absolute paths are embedded in the outputs.
