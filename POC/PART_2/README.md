# Part 2 — RFM Segmentation & Retention Strategy

This folder contains a reproducible RFM-plus-signals segmentation workflow and the business outputs required for Part 2.

## Deliverables

- `rfm_segmentation.ipynb`
- `segments.csv`
- `retention_strategy.md`
- `manual_review_cases.md`
- `charts/`
- `build_part2.py`
- `requirements.txt`

## Data Loading

The script checks these relative locations:

1. `PART_2/data/`
2. `../d2c churn data package/d2c churn data package/`

If this folder is copied into a standalone repository, place the churn CSV files in `data/`.

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_part2.py
```

## What The Workflow Does

- Builds RFM features from pre-snapshot order data.
- Adds non-RFM signals: support friction, returns, app/web engagement, campaign response, and manual CRM priority.
- Assigns every customer to a justified segment.
- Writes the exact threshold-based segment rules into the strategy document so the segmentation logic is visible outside the code.
- Writes a segment-level strategy document and ten manual-review cases with real customer IDs.

## Integrity Notes

- Only pre-snapshot order history is used for segment construction.
- `churn_next_60d` is used only to evaluate the segment definitions, not to assign them operationally.
- All file references are relative, and the folder contains no secrets or machine-specific paths.
