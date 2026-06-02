# Retention Strategy

## Segment Logic

The segmentation uses classic RFM signals first and then sharpens them with support friction, return behavior, and digital engagement:

1. `Champions`: recent, frequent, high-spend buyers with low friction.
2. `Loyal Core`: strong recent value but not quite at the champion bar.
3. `Growth Potential`: fresh customers with strong engagement but lighter spend depth.
4. `Discount-Sensitive`: customers whose recent shopping pattern leans on discounts and campaign response.
5. `Service Recovery`: customers showing service pain through tickets, negative sentiment, or returns.
6. `Dormant At-Risk`: stale recency plus very low current activity.
7. `Mixed Watchlist`: remaining customers where the signal is real but not clean enough for a heavier intervention.

## RFM Feature Construction

1. `recency_days`: days between the snapshot date (`2025-09-30`) and the customer’s latest pre-snapshot order.
2. `frequency_180d`: count of distinct pre-snapshot orders in the 180 days before the snapshot.
3. `monetary_180d`: total pre-snapshot gross spend in the 180 days before the snapshot.
4. `r_score`, `f_score`, `m_score`: quintile scores from 1 to 5, where higher is better for recency freshness, order frequency, and spend.

## Data-Driven Thresholds Used In Segmentation

| threshold_name                             |   value | used_for                                                         |
|:-------------------------------------------|--------:|:-----------------------------------------------------------------|
| High-engagement sessions cutoff            |    8    | `sessions_30d >= cutoff` in the Growth Potential rule            |
| High-engagement campaign-click cutoff      |    1    | `campaign_clicks_30d >= cutoff` in the Growth Potential rule     |
| Discount-sensitive average-discount cutoff |    0.34 | `avg_discount_pct_180d >= cutoff` in the Discount-Sensitive rule |

## Exact Segment Rules

| segment_name       | exact_rule                                                                                                                                                                                 | signals_used                                              |
|:-------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
| Champions          | Assigned if r_score >= 4, f_score >= 4, m_score >= 4, return_rate_180d <= 0.10, and ticket_count_90d <= 1.                                                                                 | RFM + returns + support complaints                        |
| Loyal Core         | Assigned if not already Champions and r_score >= 4, f_score >= 3, and m_score >= 3.                                                                                                        | RFM                                                       |
| Growth Potential   | Assigned if not matched above, r_score >= 4, high_engagement is true, and either f_score <= 2 or m_score <= 2. `high_engagement` means sessions_30d >= 8 or campaign_clicks_30d >= 1.      | RFM + app/web activity + campaign engagement              |
| Discount-Sensitive | Assigned if not matched above, avg_discount_pct_180d >= 0.34, campaign_clicks_30d > 0, and r_score >= 2.                                                                                   | Discount usage + campaign engagement + recency            |
| Service Recovery   | Assigned if not matched above, r_score >= 2, and service_friction is true. service_friction means (ticket_count_90d >= 1 and negative_ticket_rate_90d >= 0.50) or return_rate_180d > 0.25. | Support complaints + ticket sentiment + returns + recency |
| Dormant At-Risk    | Assigned if not matched above, r_score <= 2, and sessions_30d <= 2.                                                                                                                        | Recency + app/web activity                                |
| Mixed Watchlist    | Catch-all segment for customers not captured by any higher-priority rule above.                                                                                                            | Residual mixed signal set                                 |

## Segment-Level Evidence

| segment_name       |   customers |   observed_churn_pct |   avg_recency_days |   avg_frequency_180d |   avg_monetary_180d |   avg_sessions_30d |   avg_ticket_count_90d |   avg_return_rate_180d |
|:-------------------|------------:|---------------------:|-------------------:|---------------------:|--------------------:|-------------------:|-----------------------:|-----------------------:|
| Dormant At-Risk    |         427 |                 86.4 |           189.817  |             0.779859 |             597.812 |            1.01171 |             0.00234192 |             0.0163934  |
| Discount-Sensitive |         112 |                 60.7 |            85.3839 |             1.60714  |            1068.51  |            6.47321 |             0.25       |             0.077381   |
| Service Recovery   |         200 |                 56.5 |            69.48   |             1.81     |            1421.51  |            5.28    |             0.88       |             0.31625    |
| Mixed Watchlist    |         851 |                 53.9 |           104.145  |             1.34195  |             991.309 |            5.54877 |             0.039953   |             0.00705053 |
| Growth Potential   |         221 |                 21.7 |            20.0136 |             1.0362   |             685.719 |            8.29412 |             0.266968   |             0.0859729  |
| Loyal Core         |         290 |                 15.2 |            21.4172 |             2.32414  |            1611.75  |            7.15862 |             0.568966   |             0.150718   |
| Champions          |         299 |                  8.7 |            19.8328 |             3.04348  |            2413.32  |            7.55853 |             0.29097    |             0          |

## Expected Business Value

The table below is a directional estimate of where value is sitting in the portfolio. `estimated_value_at_risk_inr` is calculated as:

`customers * avg_monetary_180d * observed_churn_rate`

It is not an uplift estimate, but it does show where churn intersects with recent spend strongly enough to matter financially.

| segment_name       |   estimated_value_at_risk_inr |   priority_index | expected_business_value                                                                                |
|:-------------------|------------------------------:|-----------------:|:-------------------------------------------------------------------------------------------------------|
| Mixed Watchlist    |                        455011 |            437.7 | Broad middle pool with real revenue at risk, but action should stay lightweight until signal sharpens. |
| Dormant At-Risk    |                        220592 |             86.6 | Large at-risk pool; value depends on cheap win-back because current engagement is weak.                |
| Service Recovery   |                        160631 |            168.1 | High save potential because customers still engage, spend materially, and show fixable friction.       |
| Discount-Sensitive |                         72659 |            193.9 | Moderate churn with promotion response; recoverable value if discounting stays disciplined.            |
| Loyal Core         |                         70917 |            166.3 | Healthy repeat buyers worth preserving with low-friction nudges before cadence softens.                |
| Champions          |                         62746 |            359.2 | Low churn but very high spend; protect margin and loyalty rather than overspend on saves.              |
| Growth Potential   |                         32914 |             76.9 | Lower current spend but strong engagement; good upside from category expansion.                        |

## Recommended Actions

| segment_name       | recommended_action                                             |   estimated_cost_per_customer_inr | rationale                                                     |
|:-------------------|:---------------------------------------------------------------|----------------------------------:|:--------------------------------------------------------------|
| Champions          | VIP early-access message with zero-discount content            |                                 5 | Protect margin while rewarding the best customers.            |
| Loyal Core         | Timed replenishment reminder plus free shipping                |                                12 | Keep a healthy repeat cadence without over-discounting.       |
| Growth Potential   | Cross-category starter bundle or personalized routine builder  |                                18 | Broaden basket depth while engagement is still high.          |
| Discount-Sensitive | Minimum-basket bundle discount                                 |                                25 | Speak to price sensitivity without collapsing AOV.            |
| Service Recovery   | Agent callback plus replacement or free-shipping credit        |                                30 | Fix resolvable friction before it becomes irreversible churn. |
| Dormant At-Risk    | Win-back free-shipping reminder with product-specific creative |                                12 | Reactivate customers before inactivity hardens further.       |
| Mixed Watchlist    | Low-cost reminder or content-led nurture                       |                                 8 | Maintain contact while gathering more signal.                 |

## Budgeted Plan

Assumed campaign budget: **₹12,000**

| segment_name       |   customers_in_segment |   estimated_cost_per_customer |   targeted_customers_under_budget |   planned_spend_inr |
|:-------------------|-----------------------:|------------------------------:|----------------------------------:|--------------------:|
| Service Recovery   |                    200 |                            30 |                               200 |                6000 |
| Dormant At-Risk    |                    427 |                            12 |                               427 |                5124 |
| Discount-Sensitive |                    112 |                            25 |                                35 |                 875 |
| Growth Potential   |                    221 |                            18 |                                 0 |                   0 |
| Mixed Watchlist    |                    851 |                             8 |                                 0 |                   0 |
| Loyal Core         |                    290 |                            12 |                                 0 |                   0 |
| Champions          |                    299 |                             5 |                                 0 |                   0 |

With a budget of ₹12,000, the first segment to prioritize is **Service Recovery**. It combines a materially high churn rate (56.5%) with meaningful recent value (₹1422 average 180-day spend) and still-visible engagement (5.3 sessions). Dormant At-Risk customers churn even more heavily (86.4%), but their average engagement is far lower (1.0 sessions), which makes them a second-wave priority rather than the first rupee spent.

## Practical Guardrails

1. Do not spend discount budget on `Champions` first. Their churn is already the lowest, so a margin-light loyalty treatment is enough.
2. Treat `Service Recovery` as an operations-led retention queue, not a coupon queue.
3. For `Discount-Sensitive`, use basket-building offers instead of flat percentage discounts.
4. For `Dormant At-Risk`, suppress repeated offers after one failed win-back touch to avoid wasting spend on fully inactive customers.
