from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

MAX_TARGET_RATE = 0.30


def find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data",
        here.parent / "d2c churn data package" / "d2c churn data package",
        here.parent / "data",
    ]
    for candidate in candidates:
        if (candidate / "rfm_modeling_snapshot.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Place the churn CSV files in PART_4/data/ or in ../d2c churn data package/d2c churn data package/."
    )


def load_table(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "rfm_modeling_snapshot.csv")


def choose_threshold(y_true: pd.Series, probabilities: pd.Series) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in [round(x, 4) for x in list(pd.Series(range(5, 96)).div(100))]:
        predictions = (probabilities >= threshold).astype(int)
        positive_rate = predictions.mean()
        if positive_rate > MAX_TARGET_RATE:
            continue
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return float(best_threshold)


def main() -> None:
    root = Path(__file__).resolve().parent
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(find_data_dir())
    feature_columns = [c for c in df.columns if c not in ["customer_id", "snapshot_date", "churn_next_60d", "split"]]
    categorical_columns = df[feature_columns].select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    train = df.loc[df["split"] == "train"].copy()
    validation = df.loc[df["split"] == "validation"].copy()
    train_plus_validation = df.loc[df["split"].isin(["train", "validation"])].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    validation_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear")),
        ]
    )
    validation_model.fit(train[feature_columns], train["churn_next_60d"])
    validation_probabilities = validation_model.predict_proba(validation[feature_columns])[:, 1]
    threshold = choose_threshold(validation["churn_next_60d"], pd.Series(validation_probabilities))

    final_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear")),
        ]
    )
    final_model.fit(train_plus_validation[feature_columns], train_plus_validation["churn_next_60d"])

    artifact = {
        "model_name": "logistic_regression",
        "threshold": threshold,
        "feature_columns": feature_columns,
        "model": final_model,
    }
    joblib.dump(artifact, model_dir / "model.pkl")

    validation_predictions = (validation_probabilities >= threshold).astype(int)
    metadata = {
        "model_name": "logistic_regression",
        "threshold": threshold,
        "positive_rate_validation": round(float(validation_predictions.mean()), 4),
        "precision_validation": round(
            float(precision_score(validation["churn_next_60d"], validation_predictions, zero_division=0)), 4
        ),
        "recall_validation": round(
            float(recall_score(validation["churn_next_60d"], validation_predictions, zero_division=0)), 4
        ),
        "f1_validation": round(
            float(f1_score(validation["churn_next_60d"], validation_predictions, zero_division=0)), 4
        ),
        "feature_columns": feature_columns,
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model artifact written to {model_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
