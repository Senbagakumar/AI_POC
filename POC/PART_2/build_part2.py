from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import seaborn as sns

SNAPSHOT_DATE = pd.Timestamp("2025-09-30")
BUDGET_INR = 12000

SEGMENT_ACTIONS = {
    "Champions": {
        "action": "VIP early-access message with zero-discount content",
        "cost_per_customer": 5,
        "goal": "Protect margin while rewarding the best customers.",
    },
    "Loyal Core": {
        "action": "Timed replenishment reminder plus free shipping",
        "cost_per_customer": 12,
        "goal": "Keep a healthy repeat cadence without over-discounting.",
    },
    "Growth Potential": {
        "action": "Cross-category starter bundle or personalized routine builder",
        "cost_per_customer": 18,
        "goal": "Broaden basket depth while engagement is still high.",
    },
    "Discount-Sensitive": {
        "action": "Minimum-basket bundle discount",
        "cost_per_customer": 25,
        "goal": "Speak to price sensitivity without collapsing AOV.",
    },
    "Service Recovery": {
        "action": "Agent callback plus replacement or free-shipping credit",
        "cost_per_customer": 30,
        "goal": "Fix resolvable friction before it becomes irreversible churn.",
    },
    "Dormant At-Risk": {
        "action": "Win-back free-shipping reminder with product-specific creative",
        "cost_per_customer": 12,
        "goal": "Reactivate customers before inactivity hardens further.",
    },
    "Mixed Watchlist": {
        "action": "Low-cost reminder or content-led nurture",
        "cost_per_customer": 8,
        "goal": "Maintain contact while gathering more signal.",
    },
}


def find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data",
        here.parent / "d2c churn data package" / "d2c churn data package",
        here.parent / "data",
    ]
    for candidate in candidates:
        if (candidate / "customers.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Place the churn CSV files in PART_2/data/ or in ../d2c churn data package/d2c churn data package/."
    )


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "customers": pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"]),
        "orders": pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"]),
        "support_tickets": pd.read_csv(data_dir / "support_tickets.csv", parse_dates=["ticket_date"]),
        "web_events_snapshot": pd.read_csv(
            data_dir / "web_events_snapshot.csv", parse_dates=["snapshot_date"]
        ),
        "churn_labels": pd.read_csv(data_dir / "churn_labels.csv", parse_dates=["snapshot_date"]),
        "intervention_history": pd.read_csv(
            data_dir / "intervention_history.csv", parse_dates=["snapshot_date"]
        ),
    }


def build_segmentation_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"]
    orders = data["orders"]
    support = data["support_tickets"]
    web = data["web_events_snapshot"]
    labels = data["churn_labels"]
    interventions = data["intervention_history"]

    pre_orders = orders.loc[orders["order_date"] <= SNAPSHOT_DATE].copy()
    orders_180d = pre_orders.loc[pre_orders["order_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=180)].copy()
    tickets_90d = support.loc[support["ticket_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=90)].copy()

    table = customers[["customer_id", "signup_date", "preferred_category", "acquisition_channel"]].copy()

    last_order = pre_orders.groupby("customer_id")["order_date"].max().rename("last_order_date")
    table = table.merge(last_order, on="customer_id", how="left")
    table["recency_days"] = (SNAPSHOT_DATE - table["last_order_date"]).dt.days
    table["recency_days"] = table["recency_days"].fillna((SNAPSHOT_DATE - table["signup_date"]).dt.days + 999)

    order_agg = (
        orders_180d.groupby("customer_id")
        .agg(
            frequency_180d=("order_id", "nunique"),
            monetary_180d=("gross_amount", "sum"),
            avg_discount_pct_180d=("discount_pct", "mean"),
            return_rate_180d=("returned", "mean"),
            category_diversity_180d=("category", "nunique"),
        )
        .reset_index()
    )
    table = table.merge(order_agg, on="customer_id", how="left")

    support_agg = (
        tickets_90d.groupby("customer_id")
        .agg(
            ticket_count_90d=("ticket_id", "count"),
            negative_ticket_rate_90d=("sentiment_score", lambda s: float((s < 0).mean())),
            avg_resolution_hours_90d=("resolution_hours", "mean"),
        )
        .reset_index()
    )
    table = table.merge(support_agg, on="customer_id", how="left")

    table = table.merge(
        web[
            [
                "customer_id",
                "sessions_30d",
                "product_views_30d",
                "cart_adds_30d",
                "campaign_clicks_30d",
                "last_visit_days_ago",
            ]
        ],
        on="customer_id",
        how="left",
    )
    table = table.merge(
        interventions[
            ["customer_id", "last_campaign_received", "last_campaign_cost", "manual_priority_bucket"]
        ],
        on="customer_id",
        how="left",
    )
    table = table.merge(labels[["customer_id", "churn_next_60d"]], on="customer_id", how="left")

    fill_zero = [
        "frequency_180d",
        "monetary_180d",
        "avg_discount_pct_180d",
        "return_rate_180d",
        "category_diversity_180d",
        "ticket_count_90d",
        "negative_ticket_rate_90d",
        "avg_resolution_hours_90d",
    ]
    for column in fill_zero:
        table[column] = table[column].fillna(0)

    table["r_score"] = pd.qcut(table["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    table["f_score"] = pd.qcut(
        table["frequency_180d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    table["m_score"] = pd.qcut(
        table["monetary_180d"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    table["rfm_total"] = table["r_score"] + table["f_score"] + table["m_score"]

    sessions_q75 = table["sessions_30d"].quantile(0.75)
    clicks_q75 = table["campaign_clicks_30d"].quantile(0.75)
    discount_q75 = table["avg_discount_pct_180d"].quantile(0.75)

    table["high_engagement"] = (table["sessions_30d"] >= sessions_q75) | (
        table["campaign_clicks_30d"] >= clicks_q75
    )
    table["service_friction"] = (
        ((table["ticket_count_90d"] >= 1) & (table["negative_ticket_rate_90d"] >= 0.5))
        | (table["return_rate_180d"] > 0.25)
    )
    table["discount_sensitive_flag"] = (
        (table["avg_discount_pct_180d"] >= discount_q75) & (table["campaign_clicks_30d"] > 0)
    )

    segments = []
    for row in table.itertuples():
        if (
            row.r_score >= 4
            and row.f_score >= 4
            and row.m_score >= 4
            and row.return_rate_180d <= 0.1
            and row.ticket_count_90d <= 1
        ):
            segment = "Champions"
        elif row.r_score >= 4 and row.f_score >= 3 and row.m_score >= 3:
            segment = "Loyal Core"
        elif row.r_score >= 4 and row.high_engagement and (row.f_score <= 2 or row.m_score <= 2):
            segment = "Growth Potential"
        elif row.discount_sensitive_flag and row.r_score >= 2:
            segment = "Discount-Sensitive"
        elif row.r_score >= 2 and row.service_friction:
            segment = "Service Recovery"
        elif row.r_score <= 2 and row.sessions_30d <= 2:
            segment = "Dormant At-Risk"
        else:
            segment = "Mixed Watchlist"
        segments.append(segment)

    table["segment_name"] = segments
    table.attrs["segment_thresholds"] = {
        "sessions_q75": float(sessions_q75),
        "clicks_q75": float(clicks_q75),
        "discount_q75": float(discount_q75),
    }
    return table


def build_segment_summary(table: pd.DataFrame) -> pd.DataFrame:
    summary = (
        table.groupby("segment_name")
        .agg(
            customers=("customer_id", "count"),
            observed_churn_rate=("churn_next_60d", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency_180d=("frequency_180d", "mean"),
            avg_monetary_180d=("monetary_180d", "mean"),
            avg_sessions_30d=("sessions_30d", "mean"),
            avg_ticket_count_90d=("ticket_count_90d", "mean"),
            avg_return_rate_180d=("return_rate_180d", "mean"),
            avg_discount_pct_180d=("avg_discount_pct_180d", "mean"),
            avg_campaign_clicks_30d=("campaign_clicks_30d", "mean"),
        )
        .reset_index()
    )
    summary["observed_churn_pct"] = (summary["observed_churn_rate"] * 100).round(1)
    summary["estimated_cost_per_customer"] = summary["segment_name"].map(
        lambda segment: SEGMENT_ACTIONS[segment]["cost_per_customer"]
    )
    summary["priority_index"] = (
        summary["avg_monetary_180d"]
        * summary["observed_churn_rate"]
        * (summary["avg_sessions_30d"] + 1)
        / summary["estimated_cost_per_customer"]
    ).round(1)
    return summary.sort_values("observed_churn_rate", ascending=False)


def create_charts(summary: pd.DataFrame, charts_dir: Path) -> list[dict[str, str]]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    chart_specs: list[dict[str, str]] = []

    ordered = summary.sort_values("customers", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=ordered,
        x="segment_name",
        y="customers",
        hue="segment_name",
        palette="YlGnBu",
        legend=False,
    )
    plt.title("Customer Count by Segment")
    plt.xlabel("")
    plt.ylabel("Customers")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = charts_dir / "01_segment_counts.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Customer Count by Segment",
            "path": path.name,
            "caption": "The segmentation covers the full 2,400-customer base and keeps the largest ambiguous pool in a separate watchlist instead of forcing a false-precision label.",
        }
    )

    ordered = summary.sort_values("observed_churn_rate", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=ordered,
        x="segment_name",
        y="observed_churn_pct",
        hue="segment_name",
        palette="rocket",
        legend=False,
    )
    plt.title("Observed 60-Day Churn Rate by Segment")
    plt.xlabel("")
    plt.ylabel("Churn rate (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = charts_dir / "02_segment_churn_rates.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Observed 60-Day Churn Rate by Segment",
            "path": path.name,
            "caption": "The segment definitions are not arbitrary: churn separates clearly between Champions/Loyal Core and the risk-heavy segments.",
        }
    )

    score_frame = summary[
        ["segment_name", "avg_recency_days", "avg_frequency_180d", "avg_monetary_180d"]
    ].copy()
    score_frame = score_frame.set_index("segment_name")
    normalized = (score_frame - score_frame.min()) / (score_frame.max() - score_frame.min())
    plt.figure(figsize=(8, 5))
    sns.heatmap(normalized, annot=score_frame.round(1), cmap="viridis", fmt="")
    plt.title("Segment Feature Profile: Recency, Frequency, and Monetary")
    plt.tight_layout()
    path = charts_dir / "03_segment_rfm_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Segment Feature Profile: Recency, Frequency, and Monetary",
            "path": path.name,
            "caption": "The segments are anchored in RFM behavior first, then sharpened with support and engagement signals.",
        }
    )

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=summary,
        x="avg_monetary_180d",
        y="observed_churn_pct",
        size="customers",
        sizes=(80, 800),
        hue="segment_name",
        legend=False,
    )
    for row in summary.itertuples():
        plt.text(row.avg_monetary_180d + 5, row.observed_churn_pct + 0.4, row.segment_name, fontsize=8)
    plt.title("Segment Value at Risk: Average Monetary vs Observed Churn")
    plt.xlabel("Average 180-day spend (INR)")
    plt.ylabel("Observed churn rate (%)")
    plt.tight_layout()
    path = charts_dir / "04_segment_value_at_risk.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Segment Value at Risk: Average Monetary vs Observed Churn",
            "path": path.name,
            "caption": "This chart helps separate cold-but-low-value groups from segments where the brand still has material spend worth protecting.",
        }
    )

    return chart_specs


def build_actions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_name": segment,
                "recommended_action": payload["action"],
                "estimated_cost_per_customer_inr": payload["cost_per_customer"],
                "rationale": payload["goal"],
            }
            for segment, payload in SEGMENT_ACTIONS.items()
        ]
    )


def build_expected_value_table(summary: pd.DataFrame) -> pd.DataFrame:
    value_table = summary.copy()
    value_table["estimated_value_at_risk_inr"] = (
        value_table["customers"] * value_table["avg_monetary_180d"] * value_table["observed_churn_rate"]
    ).round(0).astype(int)
    value_table["expected_business_value"] = value_table["segment_name"].map(
        {
            "Champions": "Low churn but very high spend; protect margin and loyalty rather than overspend on saves.",
            "Loyal Core": "Healthy repeat buyers worth preserving with low-friction nudges before cadence softens.",
            "Growth Potential": "Lower current spend but strong engagement; good upside from category expansion.",
            "Discount-Sensitive": "Moderate churn with promotion response; recoverable value if discounting stays disciplined.",
            "Service Recovery": "High save potential because customers still engage, spend materially, and show fixable friction.",
            "Dormant At-Risk": "Large at-risk pool; value depends on cheap win-back because current engagement is weak.",
            "Mixed Watchlist": "Broad middle pool with real revenue at risk, but action should stay lightweight until signal sharpens.",
        }
    )
    return value_table[
        ["segment_name", "estimated_value_at_risk_inr", "priority_index", "expected_business_value"]
    ].sort_values("estimated_value_at_risk_inr", ascending=False)


def build_rule_tables(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = table.attrs.get("segment_thresholds", {})
    threshold_table = pd.DataFrame(
        [
            {
                "threshold_name": "High-engagement sessions cutoff",
                "value": f"{thresholds.get('sessions_q75', 0):.0f}",
                "used_for": "`sessions_30d >= cutoff` in the Growth Potential rule",
            },
            {
                "threshold_name": "High-engagement campaign-click cutoff",
                "value": f"{thresholds.get('clicks_q75', 0):.0f}",
                "used_for": "`campaign_clicks_30d >= cutoff` in the Growth Potential rule",
            },
            {
                "threshold_name": "Discount-sensitive average-discount cutoff",
                "value": f"{thresholds.get('discount_q75', 0):.2f}",
                "used_for": "`avg_discount_pct_180d >= cutoff` in the Discount-Sensitive rule",
            },
        ]
    )

    rule_table = pd.DataFrame(
        [
            {
                "segment_name": "Champions",
                "exact_rule": "Assigned if r_score >= 4, f_score >= 4, m_score >= 4, return_rate_180d <= 0.10, and ticket_count_90d <= 1.",
                "signals_used": "RFM + returns + support complaints",
            },
            {
                "segment_name": "Loyal Core",
                "exact_rule": "Assigned if not already Champions and r_score >= 4, f_score >= 3, and m_score >= 3.",
                "signals_used": "RFM",
            },
            {
                "segment_name": "Growth Potential",
                "exact_rule": (
                    f"Assigned if not matched above, r_score >= 4, high_engagement is true, and either f_score <= 2 or m_score <= 2. "
                    f"`high_engagement` means sessions_30d >= {thresholds.get('sessions_q75', 0):.0f} or campaign_clicks_30d >= {thresholds.get('clicks_q75', 0):.0f}."
                ),
                "signals_used": "RFM + app/web activity + campaign engagement",
            },
            {
                "segment_name": "Discount-Sensitive",
                "exact_rule": (
                    f"Assigned if not matched above, avg_discount_pct_180d >= {thresholds.get('discount_q75', 0):.2f}, "
                    "campaign_clicks_30d > 0, and r_score >= 2."
                ),
                "signals_used": "Discount usage + campaign engagement + recency",
            },
            {
                "segment_name": "Service Recovery",
                "exact_rule": (
                    "Assigned if not matched above, r_score >= 2, and service_friction is true. "
                    "service_friction means (ticket_count_90d >= 1 and negative_ticket_rate_90d >= 0.50) or return_rate_180d > 0.25."
                ),
                "signals_used": "Support complaints + ticket sentiment + returns + recency",
            },
            {
                "segment_name": "Dormant At-Risk",
                "exact_rule": "Assigned if not matched above, r_score <= 2, and sessions_30d <= 2.",
                "signals_used": "Recency + app/web activity",
            },
            {
                "segment_name": "Mixed Watchlist",
                "exact_rule": "Catch-all segment for customers not captured by any higher-priority rule above.",
                "signals_used": "Residual mixed signal set",
            },
        ]
    )
    return threshold_table, rule_table


def rank_customers_for_partial_targeting(segment_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = segment_rows.copy()
    ranked["target_priority_score"] = (
        ranked["monetary_180d"]
        - ranked["recency_days"] * 2
        + ranked["campaign_clicks_30d"] * 50
        + ranked["sessions_30d"] * 10
        - ranked["return_rate_180d"] * 150
    )
    return ranked.sort_values("target_priority_score", ascending=False)


def build_budget_plan(table: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    budget = BUDGET_INR
    working = summary.copy()
    working["estimated_total_segment_cost"] = (
        working["customers"] * working["estimated_cost_per_customer"]
    ).astype(int)
    priority_order = [
        "Service Recovery",
        "Dormant At-Risk",
        "Discount-Sensitive",
        "Growth Potential",
        "Mixed Watchlist",
        "Loyal Core",
        "Champions",
    ]
    working["priority_rank"] = working["segment_name"].map({name: i for i, name in enumerate(priority_order, start=1)})
    working = working.sort_values("priority_rank")

    rows = []
    for item in working.itertuples():
        max_reachable = budget // item.estimated_cost_per_customer
        if max_reachable <= 0:
            targeted = 0
        else:
            targeted = min(item.customers, max_reachable)
        spend = targeted * item.estimated_cost_per_customer
        rows.append(
            {
                "segment_name": item.segment_name,
                "customers_in_segment": int(item.customers),
                "estimated_cost_per_customer": int(item.estimated_cost_per_customer),
                "targeted_customers_under_budget": int(targeted),
                "planned_spend_inr": int(spend),
            }
        )
        budget -= spend

    plan = pd.DataFrame(rows)
    priority_segment = "Service Recovery"
    service_row = summary.loc[summary["segment_name"] == priority_segment].iloc[0]
    dormant_row = summary.loc[summary["segment_name"] == "Dormant At-Risk"].iloc[0]
    rationale = (
        f"With a budget of ₹{BUDGET_INR:,}, the first segment to prioritize is **{priority_segment}**. "
        f"It combines a materially high churn rate ({service_row['observed_churn_pct']:.1f}%) with meaningful recent value "
        f"(₹{service_row['avg_monetary_180d']:.0f} average 180-day spend) and still-visible engagement ({service_row['avg_sessions_30d']:.1f} sessions). "
        f"Dormant At-Risk customers churn even more heavily ({dormant_row['observed_churn_pct']:.1f}%), but their average engagement is far lower "
        f"({dormant_row['avg_sessions_30d']:.1f} sessions), which makes them a second-wave priority rather than the first rupee spent."
    )
    return plan, rationale


def build_manual_review_cases(table: pd.DataFrame) -> pd.DataFrame:
    candidates = []

    candidates.extend(
        rank_customers_for_partial_targeting(
            table.loc[
                (table["segment_name"] == "Dormant At-Risk")
                & (table["monetary_180d"] >= table["monetary_180d"].quantile(0.75))
            ]
        )
        .head(2)
        .assign(
            review_reason="High historical spend but now dormant; expensive to ignore yet already cold.",
            recommended_decision="Use a high-touch win-back with capped incentive, then suppress if there is still no response.",
        )
        .to_dict("records")
    )

    candidates.extend(
        rank_customers_for_partial_targeting(
            table.loc[
                (table["segment_name"] == "Growth Potential")
                & ((table["ticket_count_90d"] > 0) | (table["return_rate_180d"] > 0))
            ]
        )
        .head(2)
        .assign(
            review_reason="Strong recent engagement but friction exists, so a pure upsell message could backfire.",
            recommended_decision="Fix the service issue first, then cross-sell only after the customer is stable.",
        )
        .to_dict("records")
    )

    candidates.extend(
        rank_customers_for_partial_targeting(
            table.loc[
                (table["segment_name"] == "Service Recovery")
                & (table["sessions_30d"] >= table["sessions_30d"].quantile(0.75))
            ]
        )
        .head(2)
        .assign(
            review_reason="Support friction is present, but engagement remains strong enough that the customer may return without a discount.",
            recommended_decision="Give a service-led callback or replacement credit rather than a blanket coupon.",
        )
        .to_dict("records")
    )

    candidates.extend(
        rank_customers_for_partial_targeting(
            table.loc[
                (table["segment_name"] == "Discount-Sensitive")
                & (table["return_rate_180d"] == 0)
                & (table["ticket_count_90d"] == 0)
            ]
        )
        .head(2)
        .assign(
            review_reason="Looks price-sensitive, but there is no sign of operational friction or poor product fit.",
            recommended_decision="Use a minimum-basket bundle instead of a deep one-off discount.",
        )
        .to_dict("records")
    )

    boundary_pool = table.loc[table["segment_name"] == "Mixed Watchlist"].copy()
    boundary_pool["boundary_distance"] = (boundary_pool["rfm_total"] - 9).abs()
    candidates.extend(
        boundary_pool.sort_values(
            ["boundary_distance", "monetary_180d", "sessions_30d"], ascending=[True, False, False]
        )
        .head(2)
        .assign(
            review_reason="Signals conflict: middle-of-pack RFM plus enough activity to justify a human check.",
            recommended_decision="Route to CRM analyst review before spending meaningful budget.",
        )
        .to_dict("records")
    )

    cases = pd.DataFrame(candidates)[
        [
            "customer_id",
            "segment_name",
            "recency_days",
            "frequency_180d",
            "monetary_180d",
            "sessions_30d",
            "campaign_clicks_30d",
            "ticket_count_90d",
            "return_rate_180d",
            "review_reason",
            "recommended_decision",
        ]
    ].drop_duplicates(subset=["customer_id"]).head(10)
    return cases


def generate_retention_strategy(
    summary: pd.DataFrame,
    actions: pd.DataFrame,
    value_table: pd.DataFrame,
    budget_plan: pd.DataFrame,
    budget_rationale: str,
    threshold_table: pd.DataFrame,
    rule_table: pd.DataFrame,
) -> str:
    strategy = f"""# Retention Strategy

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

{threshold_table.to_markdown(index=False)}

## Exact Segment Rules

{rule_table.to_markdown(index=False)}

## Segment-Level Evidence

{summary[['segment_name', 'customers', 'observed_churn_pct', 'avg_recency_days', 'avg_frequency_180d', 'avg_monetary_180d', 'avg_sessions_30d', 'avg_ticket_count_90d', 'avg_return_rate_180d']].to_markdown(index=False)}

## Expected Business Value

The table below is a directional estimate of where value is sitting in the portfolio. `estimated_value_at_risk_inr` is calculated as:

`customers * avg_monetary_180d * observed_churn_rate`

It is not an uplift estimate, but it does show where churn intersects with recent spend strongly enough to matter financially.

{value_table.to_markdown(index=False)}

## Recommended Actions

{actions.to_markdown(index=False)}

## Budgeted Plan

Assumed campaign budget: **₹{BUDGET_INR:,}**

{budget_plan.to_markdown(index=False)}

{budget_rationale}

## Practical Guardrails

1. Do not spend discount budget on `Champions` first. Their churn is already the lowest, so a margin-light loyalty treatment is enough.
2. Treat `Service Recovery` as an operations-led retention queue, not a coupon queue.
3. For `Discount-Sensitive`, use basket-building offers instead of flat percentage discounts.
4. For `Dormant At-Risk`, suppress repeated offers after one failed win-back touch to avoid wasting spend on fully inactive customers.
"""
    return strategy


def generate_manual_review_markdown(cases: pd.DataFrame) -> str:
    return f"""# Manual Review Cases

The ten customers below are intentionally selected because the automated segment assignment is not enough on its own. Each case mixes valuable signal with a meaningful contradiction.

{cases.to_markdown(index=False)}
"""


def build_notebook(
    output_path: Path,
    dataset_summary: pd.DataFrame,
    rfm_feature_sample: pd.DataFrame,
    signal_summary: pd.DataFrame,
    rule_table: pd.DataFrame,
    assignment_sample: pd.DataFrame,
    summary: pd.DataFrame,
    strategy: str,
    manual_review: str,
    chart_specs: list[dict[str, str]],
) -> None:
    notebook = nbf.v4.new_notebook()
    chart_sections = []
    for spec in chart_specs:
        chart_sections.append(f"### {spec['title']}\n![{spec['title']}](charts/{spec['path']})\n\n{spec['caption']}")

    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Part 2 — RFM Segmentation & Retention Strategy\n"
            "This notebook is generated from `build_part2.py` and captures the segment evidence from the actual dataset run."
        ),
        nbf.v4.new_markdown_cell(
            "## Data Loading\n\n"
            "The workflow loads the raw customer, order, support, web snapshot, churn label, and intervention files from relative paths.\n\n"
            + dataset_summary.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "SNAPSHOT_DATE = pd.Timestamp('2025-09-30')\n"
            "candidates = [\n"
            "    Path('data'),\n"
            "    Path('../d2c churn data package/d2c churn data package'),\n"
            "]\n"
            "data_dir = next(path for path in candidates if (path / 'customers.csv').exists())\n"
            "customers = pd.read_csv(data_dir / 'customers.csv', parse_dates=['signup_date'])\n"
            "orders = pd.read_csv(data_dir / 'orders.csv', parse_dates=['order_date'])\n"
            "support = pd.read_csv(data_dir / 'support_tickets.csv', parse_dates=['ticket_date'])\n"
            "web = pd.read_csv(data_dir / 'web_events_snapshot.csv', parse_dates=['snapshot_date'])\n"
            "interventions = pd.read_csv(data_dir / 'intervention_history.csv', parse_dates=['snapshot_date'])\n"
            "customers.shape, orders.shape, support.shape, web.shape, interventions.shape"
        ),
        nbf.v4.new_markdown_cell(
            "## RFM Feature Creation\n\n"
            "The segmentation starts with recency, frequency, and monetary features built from pre-snapshot orders. The sample below shows the actual engineered RFM fields used in the run.\n\n"
            + rfm_feature_sample.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "pre_orders = orders.loc[orders['order_date'] <= SNAPSHOT_DATE].copy()\n"
            "orders_180d = pre_orders.loc[pre_orders['order_date'] >= SNAPSHOT_DATE - pd.Timedelta(days=180)].copy()\n\n"
            "last_order = pre_orders.groupby('customer_id')['order_date'].max().rename('last_order_date')\n"
            "rfm = customers[['customer_id', 'signup_date']].merge(last_order, on='customer_id', how='left')\n"
            "rfm['recency_days'] = (SNAPSHOT_DATE - rfm['last_order_date']).dt.days\n"
            "rfm['recency_days'] = rfm['recency_days'].fillna((SNAPSHOT_DATE - rfm['signup_date']).dt.days + 999)\n"
            "rfm = rfm.merge(\n"
            "    orders_180d.groupby('customer_id').agg(\n"
            "        frequency_180d=('order_id', 'nunique'),\n"
            "        monetary_180d=('gross_amount', 'sum'),\n"
            "    ).reset_index(),\n"
            "    on='customer_id',\n"
            "    how='left',\n"
            ")\n"
            "rfm.head()"
        ),
        nbf.v4.new_markdown_cell(
            "## Additional Behavioural / Support Signals\n\n"
            "RFM is combined with non-RFM evidence from support, returns, campaign response, and digital activity.\n\n"
            + signal_summary.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(
            "## Segmentation Logic\n\n"
            "The table below shows the exact ordered rules used to assign segments.\n\n"
            + rule_table.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            "def assign_segment(row):\n"
            "    if row.r_score >= 4 and row.f_score >= 4 and row.m_score >= 4 and row.return_rate_180d <= 0.10 and row.ticket_count_90d <= 1:\n"
            "        return 'Champions'\n"
            "    if row.r_score >= 4 and row.f_score >= 3 and row.m_score >= 3:\n"
            "        return 'Loyal Core'\n"
            "    if row.r_score >= 4 and row.high_engagement and (row.f_score <= 2 or row.m_score <= 2):\n"
            "        return 'Growth Potential'\n"
            "    if row.discount_sensitive_flag and row.r_score >= 2:\n"
            "        return 'Discount-Sensitive'\n"
            "    if row.r_score >= 2 and row.service_friction:\n"
            "        return 'Service Recovery'\n"
            "    if row.r_score <= 2 and row.sessions_30d <= 2:\n"
            "        return 'Dormant At-Risk'\n"
            "    return 'Mixed Watchlist'\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Segment Summary\n\n"
            + summary[
                [
                    "segment_name",
                    "customers",
                    "observed_churn_pct",
                    "avg_recency_days",
                    "avg_frequency_180d",
                    "avg_monetary_180d",
                ]
            ].to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell("## Visual Evidence\n\n" + "\n\n".join(chart_sections)),
        nbf.v4.new_markdown_cell(
            "## Final Segment Assignment Sample\n\n"
            "The full customer-level output is written to `segments.csv`. The sample below shows actual final assignments with the key features used in the segmentation.\n\n"
            + assignment_sample.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(strategy),
        nbf.v4.new_markdown_cell(manual_review),
        nbf.v4.new_code_cell("# Rebuild from the command line with: python build_part2.py"),
    ]
    nbf.write(notebook, output_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = find_data_dir()
    charts_dir = root / "charts"

    data = load_data(data_dir)
    table = build_segmentation_table(data)
    summary = build_segment_summary(table)
    charts = create_charts(summary, charts_dir)
    actions = build_actions_table()
    value_table = build_expected_value_table(summary)
    threshold_table, rule_table = build_rule_tables(table)
    budget_plan, budget_rationale = build_budget_plan(table, summary)
    review_cases = build_manual_review_cases(table)
    dataset_summary = pd.DataFrame(
        [
            {"dataset": "customers.csv", "rows": int(data["customers"].shape[0]), "columns": int(data["customers"].shape[1])},
            {"dataset": "orders.csv", "rows": int(data["orders"].shape[0]), "columns": int(data["orders"].shape[1])},
            {"dataset": "support_tickets.csv", "rows": int(data["support_tickets"].shape[0]), "columns": int(data["support_tickets"].shape[1])},
            {"dataset": "web_events_snapshot.csv", "rows": int(data["web_events_snapshot"].shape[0]), "columns": int(data["web_events_snapshot"].shape[1])},
            {"dataset": "churn_labels.csv", "rows": int(data["churn_labels"].shape[0]), "columns": int(data["churn_labels"].shape[1])},
            {"dataset": "intervention_history.csv", "rows": int(data["intervention_history"].shape[0]), "columns": int(data["intervention_history"].shape[1])},
        ]
    )
    rfm_feature_sample = table[
        ["customer_id", "recency_days", "frequency_180d", "monetary_180d", "r_score", "f_score", "m_score", "rfm_total"]
    ].sort_values("customer_id").head(12)
    signal_summary = summary[
        [
            "segment_name",
            "avg_sessions_30d",
            "avg_ticket_count_90d",
            "avg_return_rate_180d",
            "avg_discount_pct_180d",
            "avg_campaign_clicks_30d",
        ]
    ].copy()
    assignment_sample = (
        table[
            [
                "customer_id",
                "segment_name",
                "recency_days",
                "frequency_180d",
                "monetary_180d",
                "sessions_30d",
                "campaign_clicks_30d",
                "ticket_count_90d",
                "return_rate_180d",
            ]
        ]
        .sort_values(["segment_name", "customer_id"])
        .groupby("segment_name", group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )

    export_columns = [
        "customer_id",
        "segment_name",
        "recency_days",
        "frequency_180d",
        "monetary_180d",
        "r_score",
        "f_score",
        "m_score",
        "sessions_30d",
        "campaign_clicks_30d",
        "ticket_count_90d",
        "return_rate_180d",
        "avg_discount_pct_180d",
        "category_diversity_180d",
        "manual_priority_bucket",
    ]
    table[export_columns].sort_values("customer_id").to_csv(root / "segments.csv", index=False)

    strategy = generate_retention_strategy(
        summary,
        actions,
        value_table,
        budget_plan,
        budget_rationale,
        threshold_table,
        rule_table,
    )
    manual_review = generate_manual_review_markdown(review_cases)
    (root / "retention_strategy.md").write_text(strategy, encoding="utf-8")
    (root / "manual_review_cases.md").write_text(manual_review, encoding="utf-8")
    build_notebook(
        root / "rfm_segmentation.ipynb",
        dataset_summary,
        rfm_feature_sample,
        signal_summary,
        rule_table,
        assignment_sample,
        summary,
        strategy,
        manual_review,
        charts,
    )

    print(f"Part 2 outputs written to {root}")


if __name__ == "__main__":
    main()
