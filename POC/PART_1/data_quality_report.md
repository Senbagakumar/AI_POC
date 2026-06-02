# Data Quality Report

## Package Manifest

| file_name                           | file_type   | rows   | columns   | grain                     | note                                                                                                          |
|:------------------------------------|:------------|:-------|:----------|:--------------------------|:--------------------------------------------------------------------------------------------------------------|
| DATA_DICTIONARY.md                  | md          | —      | —         | Reference / documentation | Included in the package and reviewed for project context                                                      |
| STUDENT_FACING_PROBLEM_STATEMENT.md | md          | —      | —         | Reference / documentation | Included in the package and reviewed for project context                                                      |
| churn_labels.csv                    | csv         | 2400   | 4         | 1 row per customer        | Target table with churn label and split assignment                                                            |
| customers.csv                       | csv         | 2400   | 9         | 1 row per customer        | Raw customer profile table                                                                                    |
| intervention_history.csv            | csv         | 2400   | 5         | 1 row per customer        | Most recent pre-snapshot campaign/intervention per customer                                                   |
| orders.csv                          | csv         | 10009  | 10        | 1 row per order line      | Raw transaction table; includes post-snapshot rows used only for label construction                           |
| rfm_modeling_snapshot.csv           | csv         | 2400   | 29        | 1 row per customer        | Derived modeling table shipped in the package; inspected here but not used to derive Part 1 business findings |
| support_tickets.csv                 | csv         | 1921   | 8         | 1 row per ticket          | Raw support interaction table                                                                                 |
| web_events_snapshot.csv             | csv         | 2400   | 10        | 1 row per customer        | Raw 30-day app/web engagement snapshot                                                                        |

## Loaded Dataset Inspection

| dataset                   |   rows |   columns | primary_key   | date_columns   | sample_columns                                                                                    |
|:--------------------------|-------:|----------:|:--------------|:---------------|:--------------------------------------------------------------------------------------------------|
| customers.csv             |   2400 |         9 | customer_id   | signup_date    | customer_id, signup_date, city_tier, age_group, acquisition_channel, loyalty_tier ...             |
| orders.csv                |  10009 |        10 | order_id      | order_date     | order_id, customer_id, order_date, category, quantity, gross_amount ...                           |
| support_tickets.csv       |   1921 |         8 | ticket_id     | ticket_date    | ticket_id, customer_id, ticket_date, issue_type, support_channel, resolution_hours ...            |
| web_events_snapshot.csv   |   2400 |        10 | customer_id   | snapshot_date  | customer_id, snapshot_date, sessions_30d, product_views_30d, cart_adds_30d, wishlist_adds_30d ... |
| churn_labels.csv          |   2400 |         4 | customer_id   | snapshot_date  | customer_id, snapshot_date, churn_next_60d, split                                                 |
| rfm_modeling_snapshot.csv |   2400 |        29 | customer_id   | snapshot_date  | customer_id, snapshot_date, city_tier, age_group, acquisition_channel, loyalty_tier ...           |
| intervention_history.csv  |   2400 |         5 | customer_id   | snapshot_date  | customer_id, snapshot_date, last_campaign_received, last_campaign_cost, manual_priority_bucket    |

## Missing Values

| dataset                   | column       |   missing_rows |   missing_pct |
|:--------------------------|:-------------|---------------:|--------------:|
| customers.csv             | loyalty_tier |           1386 |          57.8 |
| rfm_modeling_snapshot.csv | loyalty_tier |           1386 |          57.8 |
| customers.csv             | skin_type    |            401 |          16.7 |
| orders.csv                | rating       |             80 |           0.8 |

## Duplicate and Duplicate-Like Records

| dataset                   |   exact_duplicate_rows |   duplicate_primary_keys |   duplicate_like_rows |
|:--------------------------|-----------------------:|-------------------------:|----------------------:|
| customers.csv             |                      0 |                        0 |                     0 |
| orders.csv                |                      0 |                        0 |                    12 |
| support_tickets.csv       |                      0 |                        0 |                     0 |
| web_events_snapshot.csv   |                      0 |                        0 |                     0 |
| churn_labels.csv          |                      0 |                        0 |                     0 |
| rfm_modeling_snapshot.csv |                      0 |                        0 |                     0 |
| intervention_history.csv  |                      0 |                        0 |                     0 |

`orders.csv` contains intentionally duplicate-like rows whose `order_id` ends with `_DUP`. Those should be removed or collapsed into their base order before any customer aggregation.

Sample duplicate-like rows:

| order_id      | base_order_id   | customer_id   | order_date          |   gross_amount |
|:--------------|:----------------|:--------------|:--------------------|---------------:|
| ORD008249_DUP | ORD008249       | CUST00153     | 2025-11-04 00:00:00 |         321.31 |
| ORD002124_DUP | ORD002124       | CUST00628     | 2025-03-18 00:00:00 |         410.04 |
| ORD002862_DUP | ORD002862       | CUST00837     | 2025-07-12 00:00:00 |         952.02 |
| ORD002916_DUP | ORD002916       | CUST00848     | 2025-09-26 00:00:00 |         547.18 |
| ORD002970_DUP | ORD002970       | CUST00869     | 2024-12-22 00:00:00 |         818.64 |
| ORD008836_DUP | ORD008836       | CUST00875     | 2025-10-23 00:00:00 |         711.2  |
| ORD003897_DUP | ORD003897       | CUST01140     | 2025-04-14 00:00:00 |         769.96 |
| ORD004577_DUP | ORD004577       | CUST01335     | 2025-02-12 00:00:00 |         533.07 |

## Join / Key Issues

| dataset                   | primary_key   |   duplicate_primary_keys | orphan_customer_ids   |   distinct_customer_ids |
|:--------------------------|:--------------|-------------------------:|:----------------------|------------------------:|
| customers.csv             | customer_id   |                        0 | —                     |                    2400 |
| orders.csv                | order_id      |                        0 | 0                     |                    2400 |
| support_tickets.csv       | ticket_id     |                        0 | 0                     |                    1247 |
| web_events_snapshot.csv   | customer_id   |                        0 | 0                     |                    2400 |
| churn_labels.csv          | customer_id   |                        0 | 0                     |                    2400 |
| rfm_modeling_snapshot.csv | customer_id   |                        0 | 0                     |                    2400 |
| intervention_history.csv  | customer_id   |                        0 | 0                     |                    2400 |

All customer-linked tables join back cleanly to the 2,400-customer universe; the main integrity risk is duplicate handling rather than orphaned IDs.

## Invalid or Unusual Values

| check                                                     |   count | recommendation                                                                                  |
|:----------------------------------------------------------|--------:|:------------------------------------------------------------------------------------------------|
| orders.rating outside 1-5                                 |       0 | Treat as invalid rating values if any appear.                                                   |
| orders.discount_pct outside 0.0-0.7                       |       0 | Clamp or investigate pricing logic if the count is non-zero.                                    |
| orders.delivery_days outside 1-11                         |       0 | Review fulfillment timestamp logic if values fall outside the documented range.                 |
| orders.gross_amount < 0                                   |       0 | Negative order values should be treated as invalid unless explicitly documented as adjustments. |
| support_tickets.sentiment_score outside -1 to 1           |       0 | Recompute or clip sentiment scores if values fall outside the scoring range.                    |
| support_tickets.resolution_hours <= 0                     |       0 | Resolution time should be positive for closed tickets.                                          |
| Negative counts in web/app activity snapshot              |       0 | Activity metrics should be non-negative; audit source event processing if not.                  |
| last_campaign_received = none but last_campaign_cost != 0 |     404 | Reset spend to 0 or audit the CRM export before ROI analysis.                                   |
| last_campaign_received != none but last_campaign_cost = 0 |     377 | Treat campaign spend as incomplete or backfill missing costs.                                   |

## Date Consistency Checks

| check                                                   |   count | recommendation                                                                     |
|:--------------------------------------------------------|--------:|:-----------------------------------------------------------------------------------|
| customers.signup_date after snapshot date               |       0 | Future-dated signups should be audited before lifecycle analysis.                  |
| orders.order_date before customer signup_date           |       0 | Transactions before signup usually indicate join or source-system timing problems. |
| support_tickets.ticket_date before customer signup_date |       0 | Support activity should not predate account creation.                              |
| support_tickets.ticket_date after snapshot date         |       0 | Ticket history should be snapshot-aligned for Part 1 and model-safe feature work.  |
| web_events_snapshot.snapshot_date != 2025-09-30         |       0 | Snapshot tables should share the same reference date.                              |
| churn_labels.snapshot_date != 2025-09-30                |       0 | Labels must align to the shared snapshot boundary.                                 |
| intervention_history.snapshot_date != 2025-09-30        |       0 | Intervention history should align to the same snapshot date.                       |
| rfm_modeling_snapshot.snapshot_date != 2025-09-30       |       0 | The derived modeling table should align to the raw-snapshot reference date.        |

## Leakage-Sensitive Columns and Rows

| dataset                   | column_or_rows                     | why_risky                                                                                         |
|:--------------------------|:-----------------------------------|:--------------------------------------------------------------------------------------------------|
| orders.csv                | Rows where order_date > 2025-09-30 | These rows occur after the modeling snapshot and can leak future purchase behavior into features. |
| churn_labels.csv          | churn_next_60d                     | This is the target label and must never be used as an input feature.                              |
| churn_labels.csv          | split                              | This is evaluation metadata, not customer behavior.                                               |
| rfm_modeling_snapshot.csv | churn_next_60d                     | Target copy embedded in the modeling snapshot.                                                    |
| rfm_modeling_snapshot.csv | split                              | Pre-assigned fold metadata; safe for evaluation only.                                             |
| rfm_modeling_snapshot.csv | snapshot_date                      | This column is constant here but should not be treated as a predictive behavior variable.         |

## Outlier Audit

| dataset             | column           |   upper_iqr_fence |   outlier_rows |     p99 |   max_value |
|:--------------------|:-----------------|------------------:|---------------:|--------:|------------:|
| orders.csv          | gross_amount     |            1619.3 |            536 | 2308.62 |     24789.4 |
| support_tickets.csv | resolution_hours |              64.9 |              9 |   59.96 |        74.6 |

Top gross-amount outliers:

| order_id   | customer_id   | order_date          | category   |   gross_amount |   discount_pct |
|:-----------|:--------------|:--------------------|:-----------|---------------:|---------------:|
| ORD006374  | CUST01868     | 2025-03-29 00:00:00 | Skin Care  |       24789.4  |           0.13 |
| ORD000701  | CUST00211     | 2024-11-27 00:00:00 | Fragrance  |       22719.5  |           0.25 |
| ORD007206  | CUST02106     | 2024-07-13 00:00:00 | Fragrance  |       15957.5  |           0.37 |
| ORD009649  | CUST01988     | 2025-10-25 00:00:00 | Fragrance  |       12312.1  |           0.04 |
| ORD004428  | CUST01295     | 2025-05-01 00:00:00 | Baby Care  |       10643.8  |           0.04 |
| ORD004650  | CUST01360     | 2024-10-09 00:00:00 | Fragrance  |        8777.2  |           0.47 |
| ORD005399  | CUST01584     | 2024-12-31 00:00:00 | Fragrance  |        8022.5  |           0.17 |
| ORD007765  | CUST02287     | 2025-06-22 00:00:00 | Fragrance  |        3746.76 |           0.08 |
| ORD000500  | CUST00159     | 2024-06-13 00:00:00 | Fragrance  |        3376.32 |           0.17 |
| ORD001120  | CUST00324     | 2024-12-30 00:00:00 | Fragrance  |        3341.27 |           0.16 |

## Treatment Recommendations

1. Deduplicate the 12 `_DUP` rows in `orders.csv` before any order-count or spend aggregation.
2. Enforce the snapshot boundary strictly: post-snapshot orders belong to label construction, not model features.
3. Keep true missingness visible for `loyalty_tier`, `skin_type`, and `rating`; encode it rather than silently dropping rows.
4. Winsorize or log-transform `gross_amount` before modelling because spend outliers are extreme enough to dominate averages.
5. Audit `intervention_history.csv` before campaign ROI work because campaign-cost fields are internally inconsistent for hundreds of customers.

## Business-Facing Readout

1. Recency is the strongest warning sign: customers with 121+ day recency churn at **89.2%** versus **11.7%** for customers who purchased in the last month.
2. Thin recent order depth matters: customers with only one order in the last 180 days churn at **61.6%**, while the 5+ order group drops to **14.8%**.
3. Low spend depth is risky: the bottom spend quartile churns at **75.7%** versus **20.7%** for the top quartile.
4. Return-heavy customers are fragile: customers with 50%+ return rates churn at **75.0%**.
5. Acquisition quality varies: Google Search customers churn at **50.4%**, materially above the **39.8%** seen in Organic acquisition.
