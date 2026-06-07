# Help

This file explains the prerequisites and the run / verification steps for `PART_1`, `PART_2`, `PART_3`, and `PART_4`.

## Prerequisites

1. Python `3.10+` is recommended.
2. `pip` must be available.
3. The dataset package must exist in one of these relative locations:
   - `./d2c churn data package/d2c churn data package/`
   - `PART_1/data/`, `PART_2/data/`, `PART_3/data/`, or `PART_4/data/` if you want to copy data into a part-local folder.
4. No API keys, credentials, or external services are required.

## Recommended Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate -> Linux
.venv\Scripts\activate --> Windows

```

Then install the requirements for the part you want to run.

## PART_1

### Install

```bash
cd PART_1
pip install -r requirements.txt

For windows
python -m pip install -r requirements.txt

if you are getting any error 
python -m pip install --upgrade pip setuptools wheel

```

### Run

```bash
python build_part1.py
```

### What To Verify

These files should exist after the run:

- `eda_audit.ipynb`
- `data_quality_report.md`
- `business_memo.md`
- `charts/`

### Quick Checks

```bash
ls
ls charts | wc -l

For windows
dir
dir charts /A:-D /B | find /c /v ""
```

Expected:

- `eda_audit.ipynb`, `data_quality_report.md`, and `business_memo.md` are present
- `charts/` contains at least `6` charts

## PART_2

### Install

```bash
cd PART_2
pip install -r requirements.txt

For windows
python -m pip install -r requirements.txt
```

### Run

```bash
python build_part2.py
```

### What To Verify

These files should exist after the run:

- `rfm_segmentation.ipynb`
- `segments.csv`
- `retention_strategy.md`
- `manual_review_cases.md`
- `charts/`

### Quick Checks

```bash
ls
head -n 5 segments.csv
wc -l segments.csv

for windows
dir
for /f "tokens=*" %i in (segments.csv) do @echo %i & set /a c+=1 & if %c%==5 goto :eof
find /c /v "" segments.csv

```

Expected:

- `segments.csv` contains `customer_id` and `segment_name`
- `segments.csv` has one row per customer plus the header
- `manual_review_cases.md` contains at least `10` customer IDs

## PART_3

### Install

```bash
cd PART_3
pip install -r requirements.txt

For windows
python -m pip install -r requirements.txt

```

### Run

```bash
python train_churn_model.py
```

### What To Verify

These files should exist after the run:

- `churn_model.ipynb`
- `model.pkl`
- `metrics.json`
- `error_analysis.md`
- `model_card.md`
- `charts/`

### Quick Checks

```bash
ls
python - <<'PY'
import json, joblib

artifact = joblib.load("model.pkl")
metrics = json.load(open("metrics.json"))

print("model_name:", artifact["model_name"])
print("threshold:", round(float(artifact["threshold"]), 4))
print("feature_count:", len(artifact["feature_columns"]))
print("selected_model:", metrics["selected_model"])
print("selected_threshold:", metrics["selected_threshold"])
print("test_f1:", metrics["test_metrics"]["f1"])
PY



for windows
dir

python -c "import json,joblib;artifact=joblib.load('model.pkl');metrics=json.load(open('metrics.json'));print('model_name:',artifact['model_name']);print('threshold:',round(float(artifact['threshold']),4));print('feature_count:',len(artifact['feature_columns']));print('selected_model:',metrics['selected_model']);print('selected_threshold:',metrics['selected_threshold']);print('test_f1:',metrics['test_metrics']['f1'])

```

Expected:

- `model.pkl` loads successfully
- `metrics.json` includes threshold and model metrics
- `error_analysis.md` contains false positive and false negative examples

## PART_4

### Install

```bash
cd PART_4
pip install -r requirements.txt

For windows
python -m pip install -r requirements.txt
```
cls

### Build The Model Artifact

```bash
python train_model.py
```

### Run The API

```bash
uvicorn app.main:app --reload --port 8000
```

### API Endpoints To Test

- `GET /health`
- `POST /predict`
- `POST /batch_predict`

### Run Automated Tests

```bash
pytest tests -q
```

Expected:

- the API test suite passes
- the service loads `model/model.pkl`

### Manual API Checks

In another terminal:

```bash
curl http://127.0.0.1:8000/health

```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city_tier": "Tier 1",
    "age_group": "18-24",
    "acquisition_channel": "Instagram",
    "loyalty_tier": "Silver",
    "preferred_category": "Makeup",
    "marketing_consent": "Yes",
    "recency_days": 107,
    "frequency_180d": 1,
    "monetary_180d": 362.73,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.23,
    "avg_rating_180d": 3.0,
    "category_diversity_180d": 1,
    "ticket_count_90d": 0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 524,
    "sessions_30d": 1,
    "product_views_30d": 4,
    "cart_adds_30d": 0,
    "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0,
    "email_opens_30d": 2,
    "campaign_clicks_30d": 0,
    "last_visit_days_ago": 20
  }'
```

## Full Run Order

If you want to validate the full submission from scratch:

```bash
cd PART_1 && pip install -r requirements.txt && python build_part1.py
cd ../PART_2 && pip install -r requirements.txt && python build_part2.py
cd ../PART_3 && pip install -r requirements.txt && python train_churn_model.py
cd ../PART_4 && pip install -r requirements.txt && python train_model.py && pytest tests -q
```

## Notes

1. `PART_1`, `PART_2`, and `PART_3` are primarily verified by regenerating outputs and checking the generated artifacts.
2. `PART_4` has the only automated test suite in the repository.
3. If you use the shared root `.venv`, you can keep reusing it across all parts and install additional part requirements into the same environment.
