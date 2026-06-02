from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")

SNAPSHOT_DATE = "2025-09-30"
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
        "Dataset not found. Place the churn CSV files in PART_3/data/ or in ../d2c churn data package/d2c churn data package/."
    )


def load_modeling_table(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "rfm_modeling_snapshot.csv")


def build_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in ["customer_id", "snapshot_date", "churn_next_60d", "split"]]
    categorical_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, categorical_cols, numeric_cols


def build_models(categorical_cols: list[str], numeric_cols: list[str]) -> dict[str, Pipeline]:
    logistic_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    hgb_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", logistic_pre),
                ("model", LogisticRegression(max_iter=5000, solver="liblinear")),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", hgb_pre),
                ("model", HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=350, random_state=42)),
            ]
        ),
    }


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "positive_rate": round(float(predictions.mean()), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    threshold_grid = np.unique(np.round(np.linspace(0.05, 0.95, 181), 4))
    rows = []
    for threshold in threshold_grid:
        metrics = compute_metrics(y_true, probabilities, float(threshold))
        rows.append(metrics)
    table = pd.DataFrame(rows)
    eligible = table.loc[table["positive_rate"] <= MAX_TARGET_RATE].copy()
    if eligible.empty:
        eligible = table.copy()
    selected = eligible.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(selected["threshold"]), table.sort_values("threshold")


def explain_case(row: pd.Series, predicted_class: int) -> str:
    reasons = []
    if row["recency_days"] > 120:
        reasons.append("very stale last order")
    elif row["recency_days"] > 60:
        reasons.append("recency already slipping")
    if row["sessions_30d"] <= 2:
        reasons.append("weak recent activity")
    if row["frequency_180d"] <= 1:
        reasons.append("only one order in the last 180 days")
    if row["return_rate_180d"] > 0.25:
        reasons.append("elevated return rate")
    if row["ticket_count_90d"] >= 1 and row["negative_ticket_rate_90d"] >= 0.5:
        reasons.append("negative support friction")
    if row["campaign_clicks_30d"] == 0 and row["email_opens_30d"] == 0:
        reasons.append("no campaign response")

    if not reasons:
        reasons.append("mixed signals")

    if predicted_class == 1:
        prefix = "Model flagged churn because of"
    else:
        prefix = "Model missed churn because stronger positive signals masked"
    return prefix + " " + ", ".join(reasons[:3]) + "."


def build_error_analysis(
    test_frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[str, pd.DataFrame]:
    frame = test_frame.copy()
    frame["predicted_probability"] = probabilities
    frame["predicted_class"] = (frame["predicted_probability"] >= threshold).astype(int)
    frame["error_type"] = np.where(
        (frame["churn_next_60d"] == 0) & (frame["predicted_class"] == 1),
        "False Positive",
        np.where(
            (frame["churn_next_60d"] == 1) & (frame["predicted_class"] == 0),
            "False Negative",
            "Correct",
        ),
    )

    false_positives = frame.loc[frame["error_type"] == "False Positive"].sort_values(
        ["predicted_probability", "monetary_180d"], ascending=[False, False]
    ).head(5)
    false_negatives = frame.loc[frame["error_type"] == "False Negative"].sort_values(
        ["monetary_180d", "predicted_probability"], ascending=[False, False]
    ).head(5)
    review = pd.concat([false_positives, false_negatives], ignore_index=True)
    review["case_note"] = review.apply(lambda row: explain_case(row, int(row["predicted_class"])), axis=1)
    review["business_risk"] = np.where(
        review["error_type"] == "False Positive",
        "Unnecessary retention spend or outreach fatigue on a customer who would have stayed anyway.",
        "Missed intervention on a real churner, creating avoidable revenue loss and lower save-rate.",
    )

    markdown = f"""# Error Analysis

The table below lists five false positives and five false negatives from the test split at the selected business threshold.

{review[['customer_id', 'error_type', 'predicted_probability', 'churn_next_60d', 'recency_days', 'frequency_180d', 'monetary_180d', 'ticket_count_90d', 'sessions_30d', 'last_visit_days_ago', 'case_note', 'business_risk']].round(4).to_markdown(index=False)}

## Interpretation

1. **False positives** are usually customers whose recency and engagement looked weak, but who still came back in the target window.
2. **False negatives** tend to be customers with one or two still-positive signals, like moderate recency or engagement, that were not enough to offset their eventual churn.
3. The remaining opportunity is less about class balance and more about modeling contradictory signals such as recent browsing plus worsening order cadence.

## Business Risk By Error Type

1. **False positive risk:** the team may waste discount budget, contact capacity, or goodwill on customers who would have stayed without intervention.
2. **False negative risk:** the team misses a real save opportunity, which can directly reduce retained revenue and hide emerging churn patterns.
"""
    return markdown, review


def build_model_card(
    comparison: pd.DataFrame,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    selected_model_name: str,
    threshold: float,
    importance: pd.DataFrame,
) -> str:
    positive_drivers = (
        importance.sort_values("coefficient", ascending=False)
        .head(5)[["feature", "coefficient"]]
        .round(4)
        .to_markdown(index=False)
    )
    negative_drivers = (
        importance.sort_values("coefficient", ascending=True)
        .head(5)[["feature", "coefficient"]]
        .round(4)
        .to_markdown(index=False)
    )
    return f"""# Model Card

## Model Details

- **Model name:** {selected_model_name}
- **Model type:** Logistic regression baseline selected as final model because it outperformed the stronger tree-based challenger on validation PR-AUC.
- **Snapshot date:** {SNAPSHOT_DATE}
- **Target:** `churn_next_60d`
- **Decision threshold:** {threshold:.2f}

## Intended Use

This model is designed for internal CRM prioritization. It should rank customers for retention review and campaign routing, not automate customer-facing decisions without human oversight.

## Model Approach

- Input table: `rfm_modeling_snapshot.csv` with behavioral, support, returns, campaign, and profile aggregates at the 2025-09-30 snapshot
- Candidate models: logistic-regression baseline and a `HistGradientBoostingClassifier` challenger
- Preprocessing: median imputation for numeric features, categorical imputation for missing labels, one-hot encoding for logistic regression, and ordinal encoding for the tree-based challenger
- Selection rule: choose the model with the best validation PR-AUC, then set the operating threshold on the validation set under the CRM capacity constraint

## Data

- Source table: `rfm_modeling_snapshot.csv`
- Universe: 2,400 customers
- Split strategy: provided `train` / `validation` / `test`
- Leakage control: `churn_next_60d`, `split`, `customer_id`, and `snapshot_date` were excluded from features

## Performance

### Validation Comparison

{comparison.to_markdown(index=False)}

### Final Operating Point

- Validation precision: {validation_metrics['precision']:.4f}
- Validation recall: {validation_metrics['recall']:.4f}
- Validation F1: {validation_metrics['f1']:.4f}
- Validation accuracy: {validation_metrics['accuracy']:.4f}
- Validation positive rate: {validation_metrics['positive_rate']:.4f}
- Test precision: {test_metrics['precision']:.4f}
- Test recall: {test_metrics['recall']:.4f}
- Test F1: {test_metrics['f1']:.4f}
- Test accuracy: {test_metrics['accuracy']:.4f}
- Test ROC-AUC: {test_metrics['roc_auc']:.4f}
- Test PR-AUC: {test_metrics['pr_auc']:.4f}

## Key Drivers

Positive coefficients raise the churn score:

{positive_drivers}

Negative coefficients lower the churn score:

{negative_drivers}

These are conditional model effects, not causal instructions. For example, `ticket_count_90d` becomes protective after controlling for recency and spend because some highly engaged customers also contact support.

## Limitations

1. The model is trained on one snapshot and may not generalize if campaign mix, pricing, or seasonality changes.
2. It sees behavioral aggregates, not raw customer intent; sudden life-cycle shifts can still be missed.
3. The score should not be interpreted as causal evidence that a discount or intervention will work.

## Ethical and Operational Risks

1. Marketing-heavy interventions may over-target paid-acquisition cohorts if scores are used without fairness review.
2. High-risk predictions can reflect service issues, not only customer disengagement, so the response should not default to discounts.
3. The model should not be used to deny service, downgrade support quality, or suppress loyal customers from legitimate help.

## When Not To Use This Model

1. Do not use it for punitive or adverse decisions such as denying support, degrading service, or removing customer benefits.
2. Do not use it as proof that a discount, outreach, or support action will cause retention; the score is predictive, not causal.
3. Do not use it when the feature snapshot is stale, upstream definitions have changed, or campaign policy has materially shifted without retraining.
4. Do not use it without human review for cases where support context or high customer value makes the intervention decision sensitive.

## Monitoring Needs

Track feature drift, prediction-rate drift, segment-wise precision, and the realized incremental retention lift from interventions triggered by the score.
"""


def build_feature_importance_chart(model: Pipeline, output_path: Path) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False)
    top = importance.head(12).copy()
    top["direction"] = np.where(top["coefficient"] >= 0, "Raises churn risk", "Lowers churn risk")

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=top.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        hue="direction",
        dodge=False,
        palette={"Raises churn risk": "#c44e52", "Lowers churn risk": "#4c72b0"},
    )
    plt.title("Top Logistic Regression Drivers")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return importance


def create_performance_charts(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("Validation ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "01_validation_roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {average_precision_score(y_true, probabilities):.3f}")
    plt.title("Validation Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_validation_pr_curve.png", dpi=160)
    plt.close()

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Validation Confusion Matrix at Selected Threshold")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(output_dir / "03_validation_confusion_matrix.png", dpi=160)
    plt.close()


def build_notebook(
    output_path: Path,
    dataset_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    leakage_table: pd.DataFrame,
    split_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    validation_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    threshold_table: pd.DataFrame,
    selected_threshold: float,
    selected_model_name: str,
    artifact_summary: pd.DataFrame,
    error_analysis: str,
    model_card: str,
) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Part 3 — Churn Prediction Model & Model Card\n"
            "This notebook is generated from `train_churn_model.py` and documents the actual run outputs."
        ),
        nbf.v4.new_markdown_cell(
            "## Data Loading\n\n"
            "The workflow loads `rfm_modeling_snapshot.csv` from a relative data directory and uses the actual run output below.\n\n"
            + dataset_summary.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "candidates = [\n"
            "    Path('data'),\n"
            "    Path('../d2c churn data package/d2c churn data package'),\n"
            "]\n"
            "data_dir = next(path for path in candidates if (path / 'rfm_modeling_snapshot.csv').exists())\n"
            "df = pd.read_csv(data_dir / 'rfm_modeling_snapshot.csv')\n"
            "df.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## Feature Preparation\n\n"
            "The target and non-feature identifiers are excluded before model training.\n\n"
            + feature_summary.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(
            "## Leakage Checks\n\n"
            "The following columns were excluded because they would leak outcome, split assignment, or row identity into the model.\n\n"
            + leakage_table.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "feature_cols = [\n"
            "    c for c in df.columns\n"
            "    if c not in ['customer_id', 'snapshot_date', 'churn_next_60d', 'split']\n"
            "]\n"
            "categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()\n"
            "numeric_cols = [c for c in feature_cols if c not in categorical_cols]\n"
            "feature_cols[:5], len(numeric_cols), len(categorical_cols)"
        ),
        nbf.v4.new_markdown_cell(
            "## Train / Validation / Test Split\n\n"
            "The provided split column is preserved exactly.\n\n"
            + split_summary.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "train = df.loc[df['split'] == 'train'].copy()\n"
            "validation = df.loc[df['split'] == 'validation'].copy()\n"
            "test = df.loc[df['split'] == 'test'].copy()\n"
            "train.shape, validation.shape, test.shape"
        ),
        nbf.v4.new_markdown_cell(
            "## Candidate Models\n\n"
            "The notebook compares a simple baseline model against a stronger tree-based challenger.\n\n"
            "1. **Baseline model:** logistic regression with scaled numeric features and one-hot encoded categoricals.\n"
            "2. **Stronger model:** histogram gradient boosting with ordinal-encoded categoricals."
        ),
        nbf.v4.new_markdown_cell(
            "## Model Comparison\n\n" + comparison.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "# Baseline: LogisticRegression\n"
            "# Stronger challenger: HistGradientBoostingClassifier\n"
            "# Both models are trained in train_churn_model.py and compared on validation PR-AUC."
        ),
        nbf.v4.new_markdown_cell(
            "## Evaluation Metrics\n\n"
            "Validation operating-point metrics:\n\n"
            + validation_summary.to_markdown(index=False)
            + "\n\nTest operating-point metrics:\n\n"
            + test_summary.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(
            "## Threshold Selection\n\n"
            f"The final model is **{selected_model_name}** and the selected threshold is **{selected_threshold:.3f}**. "
            "Thresholds were swept on the validation set, and the final operating point had to satisfy the CRM capacity rule of positive rate <= 30%.\n\n"
            "Top candidate thresholds by validation F1:\n\n"
            + threshold_table[
                ["threshold", "accuracy", "roc_auc", "pr_auc", "precision", "recall", "f1", "positive_rate", "tn", "fp", "fn", "tp"]
            ]
            .sort_values("f1", ascending=False)
            .head(10)
            .to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(
            "## Visual Diagnostics\n\n"
            "### Validation ROC Curve\n![Validation ROC](charts/01_validation_roc_curve.png)\n\n"
            "### Validation Precision-Recall Curve\n![Validation PR](charts/02_validation_pr_curve.png)\n\n"
            "### Validation Confusion Matrix\n![Validation Confusion](charts/03_validation_confusion_matrix.png)\n\n"
            "### Top Model Drivers\n![Feature Importance](charts/04_top_model_drivers.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Final Model Saving\n\n"
            "The selected pipeline is serialized to `model.pkl` together with the threshold and feature column list.\n\n"
            + artifact_summary.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "import joblib\n\n"
            "loaded = joblib.load('model.pkl')\n"
            "loaded.keys(), loaded['model_name'], loaded['threshold']"
        ),
        nbf.v4.new_markdown_cell(error_analysis),
        nbf.v4.new_markdown_cell(model_card),
        nbf.v4.new_code_cell("# Rebuild from the command line with: python train_churn_model.py"),
    ]
    nbf.write(notebook, output_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    charts_dir = root / "charts"
    data_dir = find_data_dir()
    df = load_modeling_table(data_dir)
    feature_cols, categorical_cols, numeric_cols = build_feature_sets(df)
    dataset_summary = pd.DataFrame(
        [
            {"metric": "source_table", "value": "rfm_modeling_snapshot.csv"},
            {"metric": "rows", "value": int(df.shape[0])},
            {"metric": "columns", "value": int(df.shape[1])},
            {"metric": "snapshot_date_min", "value": str(df["snapshot_date"].min())},
            {"metric": "snapshot_date_max", "value": str(df["snapshot_date"].max())},
            {"metric": "overall_churn_rate", "value": round(float(df["churn_next_60d"].mean()), 4)},
        ]
    )
    feature_summary = pd.DataFrame(
        [
            {"metric": "total_model_features", "value": len(feature_cols)},
            {"metric": "numeric_features", "value": len(numeric_cols)},
            {"metric": "categorical_features", "value": len(categorical_cols)},
            {"metric": "excluded_columns", "value": ", ".join(["customer_id", "snapshot_date", "churn_next_60d", "split"])},
        ]
    )
    leakage_table = pd.DataFrame(
        [
            {"column": "customer_id", "reason_excluded": "row identifier; would let the model memorize customers rather than learn behavior"},
            {"column": "snapshot_date", "reason_excluded": "snapshot metadata, not customer behavior; constant-time anchor for this run"},
            {"column": "split", "reason_excluded": "evaluation assignment; not available at scoring time"},
            {"column": "churn_next_60d", "reason_excluded": "future outcome target and direct leakage"},
        ]
    )

    train = df.loc[df["split"] == "train"].copy()
    validation = df.loc[df["split"] == "validation"].copy()
    test = df.loc[df["split"] == "test"].copy()
    X_train, y_train = train[feature_cols], train["churn_next_60d"]
    X_validation, y_validation = validation[feature_cols], validation["churn_next_60d"]
    X_test, y_test = test[feature_cols], test["churn_next_60d"]

    models = build_models(categorical_cols, numeric_cols)
    comparison_rows = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        comparison_rows.append(
            {
                "model_name": model_name,
                "model_family": "baseline" if model_name == "logistic_regression" else "stronger challenger",
                **compute_metrics(y_validation, validation_probabilities, 0.5),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    selected_model_name = comparison.iloc[0]["model_name"]
    selected_model = fitted_models[selected_model_name]
    split_summary = (
        df.groupby("split")
        .agg(customers=("customer_id", "count"), churn_rate=("churn_next_60d", "mean"))
        .reset_index()
    )
    split_summary["churn_rate"] = split_summary["churn_rate"].round(4)

    validation_probabilities = selected_model.predict_proba(X_validation)[:, 1]
    threshold, threshold_table = select_threshold(y_validation, validation_probabilities)
    validation_metrics = compute_metrics(y_validation, validation_probabilities, threshold)
    validation_summary = pd.DataFrame(
        [
            {"metric": "accuracy", "value": validation_metrics["accuracy"]},
            {"metric": "roc_auc", "value": validation_metrics["roc_auc"]},
            {"metric": "pr_auc", "value": validation_metrics["pr_auc"]},
            {"metric": "precision", "value": validation_metrics["precision"]},
            {"metric": "recall", "value": validation_metrics["recall"]},
            {"metric": "f1", "value": validation_metrics["f1"]},
            {"metric": "positive_rate", "value": validation_metrics["positive_rate"]},
            {"metric": "tn", "value": validation_metrics["tn"]},
            {"metric": "fp", "value": validation_metrics["fp"]},
            {"metric": "fn", "value": validation_metrics["fn"]},
            {"metric": "tp", "value": validation_metrics["tp"]},
        ]
    )

    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_probabilities, threshold)
    test_summary = pd.DataFrame(
        [
            {"metric": "accuracy", "value": test_metrics["accuracy"]},
            {"metric": "roc_auc", "value": test_metrics["roc_auc"]},
            {"metric": "pr_auc", "value": test_metrics["pr_auc"]},
            {"metric": "precision", "value": test_metrics["precision"]},
            {"metric": "recall", "value": test_metrics["recall"]},
            {"metric": "f1", "value": test_metrics["f1"]},
            {"metric": "positive_rate", "value": test_metrics["positive_rate"]},
            {"metric": "tn", "value": test_metrics["tn"]},
            {"metric": "fp", "value": test_metrics["fp"]},
            {"metric": "fn", "value": test_metrics["fn"]},
            {"metric": "tp", "value": test_metrics["tp"]},
        ]
    )

    create_performance_charts(y_validation, validation_probabilities, threshold, charts_dir)
    importance = build_feature_importance_chart(selected_model, charts_dir / "04_top_model_drivers.png")
    error_analysis_md, error_review = build_error_analysis(test[["customer_id"] + feature_cols + ["churn_next_60d"]], test_probabilities, threshold)
    model_card_md = build_model_card(
        comparison,
        validation_metrics,
        test_metrics,
        selected_model_name,
        threshold,
        importance,
    )

    artifact = {
        "model_name": selected_model_name,
        "threshold": threshold,
        "feature_columns": feature_cols,
        "model": selected_model,
    }
    joblib.dump(artifact, root / "model.pkl")
    artifact_summary = pd.DataFrame(
        [
            {"artifact_field": "path", "value": "model.pkl"},
            {"artifact_field": "model_name", "value": selected_model_name},
            {"artifact_field": "threshold", "value": round(float(threshold), 4)},
            {"artifact_field": "feature_count", "value": len(feature_cols)},
        ]
    )

    metrics_payload = {
        "snapshot_date": SNAPSHOT_DATE,
        "capacity_rule": f"Threshold chosen from validation set with max positive rate <= {MAX_TARGET_RATE:.0%}.",
        "model_comparison_validation": comparison.to_dict(orient="records"),
        "selected_model": selected_model_name,
        "selected_threshold": round(float(threshold), 4),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_positive_risk_drivers": importance.sort_values("coefficient", ascending=False)
        .head(10)[["feature", "coefficient"]]
        .round(4)
        .to_dict(orient="records"),
        "top_negative_risk_drivers": importance.sort_values("coefficient", ascending=True)
        .head(10)[["feature", "coefficient"]]
        .round(4)
        .to_dict(orient="records"),
    }
    (root / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (root / "error_analysis.md").write_text(error_analysis_md, encoding="utf-8")
    (root / "model_card.md").write_text(model_card_md, encoding="utf-8")
    build_notebook(
        root / "churn_model.ipynb",
        dataset_summary,
        feature_summary,
        leakage_table,
        split_summary,
        comparison,
        validation_summary,
        test_summary,
        threshold_table,
        threshold,
        selected_model_name,
        artifact_summary,
        error_analysis_md,
        model_card_md,
    )

    print(f"Selected model: {selected_model_name}")
    print(f"Selected threshold: {threshold:.4f}")
    print("Top error examples:")
    print(error_review[["customer_id", "error_type", "predicted_probability"]].head(10).to_string(index=False))
    print(f"Part 3 outputs written to {root}")


if __name__ == "__main__":
    main()
