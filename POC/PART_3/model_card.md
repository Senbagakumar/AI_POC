# Model Card

## Model Details

- **Model name:** logistic_regression
- **Model type:** Logistic regression baseline selected as final model because it outperformed the stronger tree-based challenger on validation PR-AUC.
- **Snapshot date:** 2025-09-30
- **Target:** `churn_next_60d`
- **Decision threshold:** 0.70

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

| model_name             | model_family        |   threshold |   accuracy |   roc_auc |   pr_auc |   precision |   recall |     f1 |   positive_rate |   tn |   fp |   fn |   tp |
|:-----------------------|:--------------------|------------:|-----------:|----------:|---------:|------------:|---------:|-------:|----------------:|-----:|-----:|-----:|-----:|
| logistic_regression    | baseline            |         0.5 |     0.8155 |    0.8827 |   0.8676 |      0.8058 |   0.7619 | 0.7832 |          0.4137 |  162 |   27 |   35 |  112 |
| hist_gradient_boosting | stronger challenger |         0.5 |     0.7976 |    0.876  |   0.8573 |      0.7687 |   0.7687 | 0.7687 |          0.4375 |  155 |   34 |   34 |  113 |

### Final Operating Point

- Validation precision: 0.8900
- Validation recall: 0.6054
- Validation F1: 0.7206
- Validation accuracy: 0.7946
- Validation positive rate: 0.2976
- Test precision: 0.8803
- Test recall: 0.6131
- Test F1: 0.7228
- Test accuracy: 0.7649
- Test ROC-AUC: 0.8845
- Test PR-AUC: 0.8778

## Key Drivers

Positive coefficients raise the churn score:

| feature                       |   coefficient |
|:------------------------------|--------------:|
| num__recency_days             |        1.7222 |
| num__return_rate_180d         |        0.344  |
| num__negative_ticket_rate_90d |        0.301  |
| num__avg_discount_pct_180d    |        0.2939 |
| num__last_visit_days_ago      |        0.2912 |

Negative coefficients lower the churn score:

| feature                           |   coefficient |
|:----------------------------------|--------------:|
| num__monetary_180d                |       -0.4321 |
| cat__preferred_category_Fragrance |       -0.3864 |
| cat__acquisition_channel_Organic  |       -0.3841 |
| num__ticket_count_90d             |       -0.3065 |
| cat__loyalty_tier_Platinum        |       -0.2839 |

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
