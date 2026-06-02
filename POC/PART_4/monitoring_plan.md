# Monitoring Plan

## What To Track

1. **Data drift**
   Compare live feature distributions against the training snapshot for recency, sessions, ticket count, return rate, and campaign engagement.
2. **Prediction distribution**
   Track the daily share of customers scoring above the decision threshold and the full probability histogram.
3. **Business outcomes**
   Measure contacted customers, accepted interventions, repeat purchase rate, and incremental retention lift against a holdout or randomized control.
4. **API health**
   Monitor request count, latency, 4xx/5xx rates, schema-validation failures, and model-load failures.
5. **Retraining triggers**
   Retrain when any of these happen:
   - the high-risk prediction rate shifts by more than 10 percentage points versus baseline
   - precision on reviewed cases drops materially for two consecutive cycles
   - product, pricing, or campaign strategy changes enough to alter customer behavior

## Responsible Use

1. Use the score to prioritize review and route interventions, not to make fully automated customer-facing decisions.
2. A high-risk score should not force a discount. Service issues, returns, or campaign fatigue may need a different response.
3. The output must never be used to deny support, reduce customer service quality, or make sensitive inferences about protected attributes.
