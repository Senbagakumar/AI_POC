from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import seaborn as sns

SNAPSHOT_DATE = pd.Timestamp("2025-09-30")

DATE_COLUMNS = {
    "customers": ["signup_date"],
    "orders": ["order_date"],
    "support_tickets": ["ticket_date"],
    "web_events_snapshot": ["snapshot_date"],
    "churn_labels": ["snapshot_date"],
    "rfm_modeling_snapshot": ["snapshot_date"],
    "intervention_history": ["snapshot_date"],
}

DATASET_GRAINS = {
    "customers": "1 row per customer",
    "orders": "1 row per order line",
    "support_tickets": "1 row per ticket",
    "web_events_snapshot": "1 row per customer",
    "churn_labels": "1 row per customer",
    "rfm_modeling_snapshot": "1 row per customer",
    "intervention_history": "1 row per customer",
}

DATASET_NOTES = {
    "customers": "Raw customer profile table",
    "orders": "Raw transaction table; includes post-snapshot rows used only for label construction",
    "support_tickets": "Raw support interaction table",
    "web_events_snapshot": "Raw 30-day app/web engagement snapshot",
    "churn_labels": "Target table with churn label and split assignment",
    "rfm_modeling_snapshot": "Derived modeling table shipped in the package; inspected here but not used to derive Part 1 business findings",
    "intervention_history": "Most recent pre-snapshot campaign/intervention per customer",
}

PRIMARY_KEYS = {
    "customers": "customer_id",
    "orders": "order_id",
    "support_tickets": "ticket_id",
    "web_events_snapshot": "customer_id",
    "churn_labels": "customer_id",
    "rfm_modeling_snapshot": "customer_id",
    "intervention_history": "customer_id",
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
        "Dataset not found. Place the churn CSV files in PART_1/data/ or in ../d2c churn data package/d2c churn data package/."
    )


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for dataset_name, parse_dates in DATE_COLUMNS.items():
        path = data_dir / f"{dataset_name}.csv"
        if path.exists():
            data[dataset_name] = pd.read_csv(path, parse_dates=parse_dates)
    return data


def build_package_manifest(data_dir: Path, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".csv":
            dataset_name = path.stem
            frame = data[dataset_name]
            rows.append(
                {
                    "file_name": path.name,
                    "file_type": "csv",
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "grain": DATASET_GRAINS.get(dataset_name, "Tabular dataset"),
                    "note": DATASET_NOTES.get(dataset_name, "Included tabular file"),
                }
            )
        else:
            rows.append(
                {
                    "file_name": path.name,
                    "file_type": path.suffix.lstrip(".") or "file",
                    "rows": "—",
                    "columns": "—",
                    "grain": "Reference / documentation",
                    "note": "Included in the package and reviewed for project context",
                }
            )
    return pd.DataFrame(rows)


def build_schema_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in data.items():
        date_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[column])
        ]
        rows.append(
            {
                "dataset": f"{dataset_name}.csv",
                "rows": len(frame),
                "columns": len(frame.columns),
                "primary_key": PRIMARY_KEYS[dataset_name],
                "date_columns": ", ".join(date_columns) if date_columns else "—",
                "sample_columns": ", ".join(frame.columns[:6])
                + (" ..." if len(frame.columns) > 6 else ""),
            }
        )
    return pd.DataFrame(rows)


def build_feature_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"].copy()
    orders = data["orders"].copy()
    support = data["support_tickets"].copy()
    web = data["web_events_snapshot"].copy()
    labels = data["churn_labels"].copy()
    interventions = data["intervention_history"].copy()

    pre_orders = orders.loc[orders["order_date"] <= SNAPSHOT_DATE].copy()
    orders_180d = pre_orders.loc[
        pre_orders["order_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=180)
    ].copy()
    tickets_90d = support.loc[
        support["ticket_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=90)
    ].copy()

    order_agg = (
        pre_orders.groupby("customer_id")
        .agg(
            last_order_date=("order_date", "max"),
            order_count_total=("order_id", "nunique"),
        )
        .reset_index()
    )
    order_agg["recency_days"] = (SNAPSHOT_DATE - order_agg["last_order_date"]).dt.days

    order_180 = (
        orders_180d.groupby("customer_id")
        .agg(
            frequency_180d=("order_id", "nunique"),
            monetary_180d=("gross_amount", "sum"),
            avg_discount_pct_180d=("discount_pct", "mean"),
            return_rate_180d=("returned", "mean"),
            avg_rating_180d=("rating", "mean"),
            category_diversity_180d=("category", "nunique"),
        )
        .reset_index()
    )

    support_90 = (
        tickets_90d.groupby("customer_id")
        .agg(
            ticket_count_90d=("ticket_id", "count"),
            negative_ticket_rate_90d=("sentiment_score", lambda s: float((s < 0).mean())),
            reopened_rate_90d=("reopened", "mean"),
            avg_resolution_hours_90d=("resolution_hours", "mean"),
        )
        .reset_index()
    )

    feature_table = (
        customers.merge(order_agg, on="customer_id", how="left")
        .merge(order_180, on="customer_id", how="left")
        .merge(web, on="customer_id", how="left")
        .merge(support_90, on="customer_id", how="left")
        .merge(interventions, on="customer_id", how="left")
        .merge(labels[["customer_id", "churn_next_60d", "split"]], on="customer_id", how="left")
    )

    zero_fill_columns = [
        "frequency_180d",
        "monetary_180d",
        "avg_discount_pct_180d",
        "return_rate_180d",
        "avg_rating_180d",
        "category_diversity_180d",
        "ticket_count_90d",
        "negative_ticket_rate_90d",
        "reopened_rate_90d",
        "avg_resolution_hours_90d",
    ]
    for column in zero_fill_columns:
        feature_table[column] = feature_table[column].fillna(0)

    feature_table["recency_days"] = feature_table["recency_days"].fillna(
        (SNAPSHOT_DATE - feature_table["signup_date"]).dt.days + 999
    )
    feature_table["loyalty_tier"] = feature_table["loyalty_tier"].fillna("None")
    feature_table["skin_type"] = feature_table["skin_type"].fillna("Unknown")
    feature_table["last_campaign_received"] = feature_table["last_campaign_received"].fillna("unknown")
    feature_table["manual_priority_bucket"] = feature_table["manual_priority_bucket"].fillna("unknown")
    return feature_table


def add_buckets(feature_table: pd.DataFrame) -> pd.DataFrame:
    frame = feature_table.copy()
    frame["recency_bucket"] = pd.cut(
        frame["recency_days"],
        bins=[-1, 30, 60, 120, frame["recency_days"].max()],
        labels=["0-30", "31-60", "61-120", "121+"],
    )
    frame["sessions_bucket"] = pd.cut(
        frame["sessions_30d"],
        bins=[-1, 2, 5, 8, frame["sessions_30d"].max()],
        labels=["0-2", "3-5", "6-8", "9+"],
    )
    frame["return_rate_bucket"] = pd.cut(
        frame["return_rate_180d"],
        bins=[-0.01, 0, 0.25, 0.5, 1.0],
        labels=["0", "0-25%", "25-50%", "50%+"],
    )
    frame["category_diversity_bucket"] = pd.cut(
        frame["category_diversity_180d"],
        bins=[-0.01, 1, 2, 3, frame["category_diversity_180d"].max()],
        labels=["1", "2", "3", "4+"],
    )
    frame["reopened_bucket"] = pd.cut(
        frame["reopened_rate_90d"],
        bins=[-0.01, 0, 0.5, 1.0],
        labels=["0%", "1-50%", "50%+"],
    )
    frame["monetary_bucket"] = pd.qcut(
        frame["monetary_180d"].rank(method="first"),
        4,
        labels=["Q1 Low", "Q2", "Q3", "Q4 High"],
    )

    frequency_bucket = pd.Series("5+", index=frame.index)
    frequency_bucket.loc[frame["frequency_180d"] <= 1] = "1"
    frequency_bucket.loc[frame["frequency_180d"] == 2] = "2"
    frequency_bucket.loc[frame["frequency_180d"].between(3, 4)] = "3-4"
    frequency_bucket.loc[frame["frequency_180d"] >= 5] = "5+"
    frame["frequency_bucket"] = pd.Categorical(
        frequency_bucket,
        categories=["1", "2", "3-4", "5+"],
        ordered=True,
    )
    return frame


def grouped_rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        frame.groupby(column, observed=False)["churn_next_60d"]
        .agg(customers="count", churn_rate="mean")
        .reset_index()
    )
    result["churn_rate_pct"] = (result["churn_rate"] * 100).round(1)
    return result


def support_issue_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    support = data["support_tickets"].copy()
    labels = data["churn_labels"][["customer_id", "churn_next_60d"]].copy()

    customer_issue = support[["customer_id", "issue_type"]].drop_duplicates()
    issue_customer = (
        customer_issue.merge(labels, on="customer_id", how="left")
        .groupby("issue_type")
        .agg(customers_with_issue=("customer_id", "nunique"), customer_churn_rate=("churn_next_60d", "mean"))
        .reset_index()
    )
    issue_ticket = (
        support.groupby("issue_type")
        .agg(
            ticket_count=("ticket_id", "count"),
            avg_resolution_hours=("resolution_hours", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            reopened_rate=("reopened", "mean"),
        )
        .reset_index()
    )

    summary = issue_customer.merge(issue_ticket, on="issue_type", how="left").sort_values(
        "customer_churn_rate", ascending=False
    )
    summary["customer_churn_pct"] = (summary["customer_churn_rate"] * 100).round(1)
    summary["reopened_pct"] = (summary["reopened_rate"] * 100).round(1)
    return summary


def campaign_summary(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    interventions = data["intervention_history"].copy()
    labels = data["churn_labels"][["customer_id", "churn_next_60d"]].copy()
    frame = interventions.merge(labels, on="customer_id", how="left")

    last_campaign = (
        frame.groupby("last_campaign_received")
        .agg(
            customers=("customer_id", "nunique"),
            avg_campaign_cost=("last_campaign_cost", "mean"),
            churn_rate=("churn_next_60d", "mean"),
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    last_campaign["churn_rate_pct"] = (last_campaign["churn_rate"] * 100).round(1)
    last_campaign["avg_campaign_cost"] = last_campaign["avg_campaign_cost"].round(1)

    manual_priority = (
        frame.groupby("manual_priority_bucket")
        .agg(
            customers=("customer_id", "nunique"),
            avg_campaign_cost=("last_campaign_cost", "mean"),
            churn_rate=("churn_next_60d", "mean"),
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    manual_priority["churn_rate_pct"] = (manual_priority["churn_rate"] * 100).round(1)
    manual_priority["avg_campaign_cost"] = manual_priority["avg_campaign_cost"].round(1)
    return last_campaign, manual_priority


def save_rate_chart(
    data: pd.DataFrame,
    x: str,
    title: str,
    output_path: Path,
    ylabel: str = "Churn rate (%)",
    palette: str = "crest",
    rotate: bool = False,
) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=data, x=x, y="churn_rate_pct", hue=x, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=20, ha="right")
    plt.ylim(0, max(100, data["churn_rate_pct"].max() + 10))
    for index, value in enumerate(data["churn_rate_pct"]):
        plt.text(index, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_charts(data: dict[str, pd.DataFrame], feature_table: pd.DataFrame, charts_dir: Path) -> list[dict[str, str]]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    for stale_chart in charts_dir.glob("*.png"):
        stale_chart.unlink()
    sns.set_theme(style="whitegrid")

    labels = data["churn_labels"].copy()
    orders = data["orders"].copy()
    frame = add_buckets(feature_table)
    issue_summary = support_issue_summary(data)
    last_campaign, _ = campaign_summary(data)

    chart_specs: list[dict[str, str]] = []

    churn_counts = (
        labels["churn_next_60d"]
        .map({0: "Retained in next 60d", 1: "Churned in next 60d"})
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="customers")
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=churn_counts,
        x="outcome",
        y="customers",
        hue="outcome",
        palette="Blues_d",
        legend=False,
    )
    plt.title("Observed 60-Day Churn Distribution")
    plt.xlabel("")
    plt.ylabel("Customers")
    for index, value in enumerate(churn_counts["customers"]):
        plt.text(index, value + 10, f"{value:,}", ha="center", fontsize=9)
    plt.tight_layout()
    path = charts_dir / "01_churn_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Observed 60-Day Churn Distribution",
            "path": path.name,
            "caption": (
                f"Churn affects {labels['churn_next_60d'].mean() * 100:.1f}% of the 2,400-customer base, "
                "so the company has a real retention problem rather than a fringe outlier cohort."
            ),
        }
    )

    monthly_orders = (
        orders.assign(
            order_month=orders["order_date"].dt.to_period("M").dt.to_timestamp(),
            period_flag=orders["order_date"].le(SNAPSHOT_DATE).map(
                {True: "Pre-snapshot orders", False: "Post-snapshot label window orders"}
            ),
        )
        .groupby(["order_month", "period_flag"], as_index=False)
        .size()
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_orders, x="order_month", y="size", hue="period_flag", marker="o")
    plt.axvline(SNAPSHOT_DATE, color="red", linestyle="--", linewidth=1.5, label="Snapshot date")
    plt.title("Monthly Order Volume and the Leakage Boundary")
    plt.xlabel("")
    plt.ylabel("Order lines")
    plt.tight_layout()
    path = charts_dir / "02_monthly_orders_and_snapshot.png"
    plt.savefig(path, dpi=160)
    plt.close()
    chart_specs.append(
        {
            "title": "Monthly Order Volume and the Leakage Boundary",
            "path": path.name,
            "caption": (
                f"The package contains {(orders['order_date'] > SNAPSHOT_DATE).sum():,} post-snapshot order rows. "
                "They help explain label construction but cannot be used as model features."
            ),
        }
    )

    acquisition_rates = grouped_rates(frame, "acquisition_channel").sort_values("churn_rate_pct", ascending=False)
    path = charts_dir / "03_acquisition_channel_vs_churn.png"
    save_rate_chart(
        acquisition_rates,
        "acquisition_channel",
        "Churn Rate by Acquisition Channel",
        path,
        rotate=True,
        palette="flare",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Acquisition Channel",
            "path": path.name,
            "caption": (
                f"Google Search customers churn at {acquisition_rates.loc[acquisition_rates['acquisition_channel'] == 'Google Search', 'churn_rate_pct'].iloc[0]:.1f}% "
                f"versus {acquisition_rates.loc[acquisition_rates['acquisition_channel'] == 'Organic', 'churn_rate_pct'].iloc[0]:.1f}% for Organic customers."
            ),
        }
    )

    frequency_rates = grouped_rates(frame, "frequency_bucket")
    path = charts_dir / "04_frequency_vs_churn.png"
    save_rate_chart(
        frequency_rates,
        "frequency_bucket",
        "Churn Rate by Order Frequency in the Last 180 Days",
        path,
        palette="mako",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Order Frequency in the Last 180 Days",
            "path": path.name,
            "caption": (
                f"Customers with only one recent order churn at {frequency_rates.loc[frequency_rates['frequency_bucket'] == '1', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"while customers with 5+ recent orders fall to {frequency_rates.loc[frequency_rates['frequency_bucket'] == '5+', 'churn_rate_pct'].iloc[0]:.1f}%."
            ),
        }
    )

    monetary_rates = grouped_rates(frame, "monetary_bucket")
    path = charts_dir / "05_monetary_vs_churn.png"
    save_rate_chart(
        monetary_rates,
        "monetary_bucket",
        "Churn Rate by 180-Day Spend Quartile",
        path,
        palette="rocket",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by 180-Day Spend Quartile",
            "path": path.name,
            "caption": (
                f"The lowest spend quartile churns at {monetary_rates.loc[monetary_rates['monetary_bucket'] == 'Q1 Low', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"compared with {monetary_rates.loc[monetary_rates['monetary_bucket'] == 'Q4 High', 'churn_rate_pct'].iloc[0]:.1f}% for the top quartile."
            ),
        }
    )

    issue_chart = issue_summary[["issue_type", "customer_churn_pct"]].rename(
        columns={"customer_churn_pct": "churn_rate_pct"}
    )
    path = charts_dir / "06_support_issue_vs_churn.png"
    save_rate_chart(
        issue_chart,
        "issue_type",
        "Customer Churn Rate by Support Issue Type",
        path,
        rotate=True,
        palette="crest",
    )
    chart_specs.append(
        {
            "title": "Customer Churn Rate by Support Issue Type",
            "path": path.name,
            "caption": (
                f"`{issue_summary.iloc[0]['issue_type']}` is the highest-churn support issue cohort at "
                f"{issue_summary.iloc[0]['customer_churn_pct']:.1f}%. This chart is customer-level, so one customer can appear in more than one issue group."
            ),
        }
    )

    return_rates = grouped_rates(frame, "return_rate_bucket")
    path = charts_dir / "07_return_rate_vs_churn.png"
    save_rate_chart(
        return_rates,
        "return_rate_bucket",
        "Churn Rate by Return Rate in the Last 180 Days",
        path,
        palette="Reds",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Return Rate in the Last 180 Days",
            "path": path.name,
            "caption": (
                f"Customers with return rates above 50% churn at {return_rates.loc[return_rates['return_rate_bucket'] == '50%+', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"versus {return_rates.loc[return_rates['return_rate_bucket'] == '0', 'churn_rate_pct'].iloc[0]:.1f}% for customers with no recent returns."
            ),
        }
    )

    sessions_rates = grouped_rates(frame, "sessions_bucket")
    path = charts_dir / "08_sessions_vs_churn.png"
    save_rate_chart(
        sessions_rates,
        "sessions_bucket",
        "Churn Rate by 30-Day Session Activity",
        path,
        palette="viridis",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by 30-Day Session Activity",
            "path": path.name,
            "caption": (
                f"Customers with only 0-2 sessions churn at {sessions_rates.loc[sessions_rates['sessions_bucket'] == '0-2', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"while customers with 9+ sessions drop to {sessions_rates.loc[sessions_rates['sessions_bucket'] == '9+', 'churn_rate_pct'].iloc[0]:.1f}%."
            ),
        }
    )

    campaign_rates = last_campaign[["last_campaign_received", "churn_rate_pct"]]
    path = charts_dir / "09_campaign_history_vs_churn.png"
    save_rate_chart(
        campaign_rates,
        "last_campaign_received",
        "Churn Rate by Most Recent Campaign Received",
        path,
        rotate=True,
        palette="cubehelix",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Most Recent Campaign Received",
            "path": path.name,
            "caption": (
                f"`new_launch` recipients churn at {last_campaign.loc[last_campaign['last_campaign_received'] == 'new_launch', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"versus {last_campaign.loc[last_campaign['last_campaign_received'] == 'none', 'churn_rate_pct'].iloc[0]:.1f}% for customers with no recent campaign."
            ),
        }
    )

    recency_rates = grouped_rates(frame, "recency_bucket")
    path = charts_dir / "10_recency_vs_churn.png"
    save_rate_chart(
        recency_rates,
        "recency_bucket",
        "Churn Rate by Days Since Last Order",
        path,
        palette="magma",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Days Since Last Order",
            "path": path.name,
            "caption": (
                f"Customers with recency over 120 days churn at {recency_rates.loc[recency_rates['recency_bucket'] == '121+', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"versus {recency_rates.loc[recency_rates['recency_bucket'] == '0-30', 'churn_rate_pct'].iloc[0]:.1f}% for customers who purchased in the last month."
            ),
        }
    )

    diversity_rates = grouped_rates(frame, "category_diversity_bucket")
    path = charts_dir / "11_category_diversity_vs_churn.png"
    save_rate_chart(
        diversity_rates,
        "category_diversity_bucket",
        "Churn Rate by Category Diversity in the Last 180 Days",
        path,
        palette="crest",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Category Diversity in the Last 180 Days",
            "path": path.name,
            "caption": (
                f"One-category shoppers churn at {diversity_rates.loc[diversity_rates['category_diversity_bucket'] == '1', 'churn_rate_pct'].iloc[0]:.1f}%, "
                f"while customers buying across three categories drop to {diversity_rates.loc[diversity_rates['category_diversity_bucket'] == '3', 'churn_rate_pct'].iloc[0]:.1f}%."
            ),
        }
    )

    ticketed = frame.loc[frame["ticket_count_90d"] > 0].copy()
    reopened_rates = grouped_rates(ticketed, "reopened_bucket")
    path = charts_dir / "12_reopened_tickets_vs_churn.png"
    save_rate_chart(
        reopened_rates,
        "reopened_bucket",
        "Churn Rate by Reopened Ticket Share (Ticketed Customers Only)",
        path,
        palette="coolwarm",
    )
    chart_specs.append(
        {
            "title": "Churn Rate by Reopened Ticket Share (Ticketed Customers Only)",
            "path": path.name,
            "caption": (
                f"Among customers who raised support tickets, those with 50%+ reopened tickets churn at "
                f"{reopened_rates.loc[reopened_rates['reopened_bucket'] == '50%+', 'churn_rate_pct'].iloc[0]:.1f}%."
            ),
        }
    )

    return chart_specs


def build_duplicate_summary(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    orders = data["orders"]
    duplicate_like = orders.loc[orders["order_id"].str.endswith("_DUP")].copy()
    duplicate_like["base_order_id"] = duplicate_like["order_id"].str.replace("_DUP", "", regex=False)
    duplicate_samples = duplicate_like[
        ["order_id", "base_order_id", "customer_id", "order_date", "gross_amount"]
    ].head(8)

    for dataset_name, frame in data.items():
        key = PRIMARY_KEYS[dataset_name]
        rows.append(
            {
                "dataset": f"{dataset_name}.csv",
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "duplicate_primary_keys": int(frame[key].duplicated().sum()),
                "duplicate_like_rows": int(duplicate_like.shape[0]) if dataset_name == "orders" else 0,
            }
        )
    return pd.DataFrame(rows), duplicate_samples


def build_key_health(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"]
    customer_ids = customers["customer_id"]
    rows = []
    for dataset_name, frame in data.items():
        key = PRIMARY_KEYS[dataset_name]
        orphan_count = "—"
        if "customer_id" in frame.columns and dataset_name != "customers":
            orphan_count = int((~frame["customer_id"].isin(customer_ids)).sum())

        rows.append(
            {
                "dataset": f"{dataset_name}.csv",
                "primary_key": key,
                "duplicate_primary_keys": int(frame[key].duplicated().sum()),
                "orphan_customer_ids": orphan_count,
                "distinct_customer_ids": int(frame["customer_id"].nunique()) if "customer_id" in frame.columns else "—",
            }
        )
    return pd.DataFrame(rows)


def build_join_summary(data: dict[str, pd.DataFrame], feature_table: pd.DataFrame) -> pd.DataFrame:
    customers = data["customers"]
    orders = data["orders"]
    support = data["support_tickets"]
    web = data["web_events_snapshot"]
    labels = data["churn_labels"]
    interventions = data["intervention_history"]

    pre_orders = orders.loc[orders["order_date"] <= SNAPSHOT_DATE].copy()
    orders_180d = pre_orders.loc[
        pre_orders["order_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=180)
    ].copy()
    tickets_90d = support.loc[
        support["ticket_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=90)
    ].copy()

    join_steps = [
        {
            "step": "Base customer universe",
            "join_key": "customer_id",
            "source_rows": int(customers.shape[0]),
            "distinct_customers": int(customers["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].nunique()),
            "coverage_pct": 100.0,
            "note": "Starting point for all customer-level analysis",
        },
        {
            "step": "Latest pre-snapshot orders",
            "join_key": "customer_id",
            "source_rows": int(pre_orders.shape[0]),
            "distinct_customers": int(pre_orders["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(pre_orders["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(pre_orders["customer_id"]).mean() * 100), 1),
            "note": "Used for total order depth and recency",
        },
        {
            "step": "Orders in last 180 days",
            "join_key": "customer_id",
            "source_rows": int(orders_180d.shape[0]),
            "distinct_customers": int(orders_180d["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(orders_180d["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(orders_180d["customer_id"]).mean() * 100), 1),
            "note": "Used for frequency, monetary value, discounts, returns, and category diversity",
        },
        {
            "step": "30-day web/app snapshot",
            "join_key": "customer_id",
            "source_rows": int(web.shape[0]),
            "distinct_customers": int(web["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(web["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(web["customer_id"]).mean() * 100), 1),
            "note": "One row per customer; clean left join",
        },
        {
            "step": "Support tickets in last 90 days",
            "join_key": "customer_id",
            "source_rows": int(tickets_90d.shape[0]),
            "distinct_customers": int(tickets_90d["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(tickets_90d["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(tickets_90d["customer_id"]).mean() * 100), 1),
            "note": "Sparse join; non-ticket customers are kept and later zero-filled",
        },
        {
            "step": "Intervention history snapshot",
            "join_key": "customer_id",
            "source_rows": int(interventions.shape[0]),
            "distinct_customers": int(interventions["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(interventions["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(interventions["customer_id"]).mean() * 100), 1),
            "note": "Campaign and manual-priority context",
        },
        {
            "step": "Observed churn labels",
            "join_key": "customer_id",
            "source_rows": int(labels.shape[0]),
            "distinct_customers": int(labels["customer_id"].nunique()),
            "matched_customers": int(customers["customer_id"].isin(labels["customer_id"]).sum()),
            "coverage_pct": round(float(customers["customer_id"].isin(labels["customer_id"]).mean() * 100), 1),
            "note": "Joined only for analysis outputs, not as a model feature",
        },
        {
            "step": "Final joined feature table",
            "join_key": "customer_id",
            "source_rows": int(feature_table.shape[0]),
            "distinct_customers": int(feature_table["customer_id"].nunique()),
            "matched_customers": int(feature_table["customer_id"].nunique()),
            "coverage_pct": round(float(feature_table["customer_id"].nunique() / customers["customer_id"].nunique() * 100), 1),
            "note": "Customer-level analysis table used for EDA and churn-pattern slicing",
        },
    ]
    return pd.DataFrame(join_steps)


def build_invalid_checks(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = data["orders"]
    support = data["support_tickets"]
    web = data["web_events_snapshot"]
    interventions = data["intervention_history"]
    activity_columns = [
        "sessions_30d",
        "product_views_30d",
        "cart_adds_30d",
        "wishlist_adds_30d",
        "abandoned_carts_30d",
        "email_opens_30d",
        "campaign_clicks_30d",
        "last_visit_days_ago",
    ]
    return pd.DataFrame(
        [
            {
                "check": "orders.rating outside 1-5",
                "count": int((~orders["rating"].dropna().between(1, 5)).sum()),
                "recommendation": "Treat as invalid rating values if any appear.",
            },
            {
                "check": "orders.discount_pct outside 0.0-0.7",
                "count": int((~orders["discount_pct"].between(0, 0.7)).sum()),
                "recommendation": "Clamp or investigate pricing logic if the count is non-zero.",
            },
            {
                "check": "orders.delivery_days outside 1-11",
                "count": int((~orders["delivery_days"].between(1, 11)).sum()),
                "recommendation": "Review fulfillment timestamp logic if values fall outside the documented range.",
            },
            {
                "check": "orders.gross_amount < 0",
                "count": int((orders["gross_amount"] < 0).sum()),
                "recommendation": "Negative order values should be treated as invalid unless explicitly documented as adjustments.",
            },
            {
                "check": "support_tickets.sentiment_score outside -1 to 1",
                "count": int((~support["sentiment_score"].between(-1, 1)).sum()),
                "recommendation": "Recompute or clip sentiment scores if values fall outside the scoring range.",
            },
            {
                "check": "support_tickets.resolution_hours <= 0",
                "count": int((support["resolution_hours"] <= 0).sum()),
                "recommendation": "Resolution time should be positive for closed tickets.",
            },
            {
                "check": "Negative counts in web/app activity snapshot",
                "count": int((web[activity_columns] < 0).sum().sum()),
                "recommendation": "Activity metrics should be non-negative; audit source event processing if not.",
            },
            {
                "check": "last_campaign_received = none but last_campaign_cost != 0",
                "count": int(
                    (
                        (interventions["last_campaign_received"] == "none")
                        & (interventions["last_campaign_cost"] != 0)
                    ).sum()
                ),
                "recommendation": "Reset spend to 0 or audit the CRM export before ROI analysis.",
            },
            {
                "check": "last_campaign_received != none but last_campaign_cost = 0",
                "count": int(
                    (
                        (interventions["last_campaign_received"] != "none")
                        & (interventions["last_campaign_cost"] == 0)
                    ).sum()
                ),
                "recommendation": "Treat campaign spend as incomplete or backfill missing costs.",
            },
        ]
    )


def build_date_consistency_checks(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["customers"]
    orders = data["orders"].merge(customers[["customer_id", "signup_date"]], on="customer_id", how="left")
    support = data["support_tickets"].merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )
    web = data["web_events_snapshot"]
    labels = data["churn_labels"]
    rfm_snapshot = data["rfm_modeling_snapshot"]
    interventions = data["intervention_history"]

    return pd.DataFrame(
        [
            {
                "check": "customers.signup_date after snapshot date",
                "count": int((customers["signup_date"] > SNAPSHOT_DATE).sum()),
                "recommendation": "Future-dated signups should be audited before lifecycle analysis.",
            },
            {
                "check": "orders.order_date before customer signup_date",
                "count": int((orders["order_date"] < orders["signup_date"]).sum()),
                "recommendation": "Transactions before signup usually indicate join or source-system timing problems.",
            },
            {
                "check": "support_tickets.ticket_date before customer signup_date",
                "count": int((support["ticket_date"] < support["signup_date"]).sum()),
                "recommendation": "Support activity should not predate account creation.",
            },
            {
                "check": "support_tickets.ticket_date after snapshot date",
                "count": int((support["ticket_date"] > SNAPSHOT_DATE).sum()),
                "recommendation": "Ticket history should be snapshot-aligned for Part 1 and model-safe feature work.",
            },
            {
                "check": "web_events_snapshot.snapshot_date != 2025-09-30",
                "count": int((web["snapshot_date"] != SNAPSHOT_DATE).sum()),
                "recommendation": "Snapshot tables should share the same reference date.",
            },
            {
                "check": "churn_labels.snapshot_date != 2025-09-30",
                "count": int((labels["snapshot_date"] != SNAPSHOT_DATE).sum()),
                "recommendation": "Labels must align to the shared snapshot boundary.",
            },
            {
                "check": "intervention_history.snapshot_date != 2025-09-30",
                "count": int((interventions["snapshot_date"] != SNAPSHOT_DATE).sum()),
                "recommendation": "Intervention history should align to the same snapshot date.",
            },
            {
                "check": "rfm_modeling_snapshot.snapshot_date != 2025-09-30",
                "count": int((rfm_snapshot["snapshot_date"] != SNAPSHOT_DATE).sum()),
                "recommendation": "The derived modeling table should align to the raw-snapshot reference date.",
            },
        ]
    )


def build_leakage_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "orders.csv",
                "column_or_rows": "Rows where order_date > 2025-09-30",
                "why_risky": "These rows occur after the modeling snapshot and can leak future purchase behavior into features.",
            },
            {
                "dataset": "churn_labels.csv",
                "column_or_rows": "churn_next_60d",
                "why_risky": "This is the target label and must never be used as an input feature.",
            },
            {
                "dataset": "churn_labels.csv",
                "column_or_rows": "split",
                "why_risky": "This is evaluation metadata, not customer behavior.",
            },
            {
                "dataset": "rfm_modeling_snapshot.csv",
                "column_or_rows": "churn_next_60d",
                "why_risky": "Target copy embedded in the modeling snapshot.",
            },
            {
                "dataset": "rfm_modeling_snapshot.csv",
                "column_or_rows": "split",
                "why_risky": "Pre-assigned fold metadata; safe for evaluation only.",
            },
            {
                "dataset": "rfm_modeling_snapshot.csv",
                "column_or_rows": "snapshot_date",
                "why_risky": "This column is constant here but should not be treated as a predictive behavior variable.",
            },
        ]
    )


def build_outlier_tables(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = data["orders"]
    support = data["support_tickets"]

    rows = []
    for dataset_name, column in [("orders", "gross_amount"), ("support_tickets", "resolution_hours")]:
        series = data[dataset_name][column]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        rows.append(
            {
                "dataset": f"{dataset_name}.csv",
                "column": column,
                "upper_iqr_fence": round(float(upper_fence), 2),
                "outlier_rows": int((series > upper_fence).sum()),
                "p99": round(float(series.quantile(0.99)), 2),
                "max_value": round(float(series.max()), 2),
            }
        )

    q1, q3 = orders["gross_amount"].quantile([0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    top_order_outliers = (
        orders.loc[orders["gross_amount"] > upper_fence]
        .sort_values("gross_amount", ascending=False)[
            ["order_id", "customer_id", "order_date", "category", "gross_amount", "discount_pct"]
        ]
        .head(10)
    )
    return pd.DataFrame(rows), top_order_outliers


def generate_quality_report(
    data_dir: Path,
    data: dict[str, pd.DataFrame],
    feature_table: pd.DataFrame,
) -> str:
    package_manifest = build_package_manifest(data_dir, data)
    schema_summary = build_schema_summary(data)
    duplicate_summary, duplicate_samples = build_duplicate_summary(data)
    key_health = build_key_health(data)
    invalid_checks = build_invalid_checks(data)
    date_checks = build_date_consistency_checks(data)
    leakage_table = build_leakage_table()
    outlier_summary, top_order_outliers = build_outlier_tables(data)

    missing_rows = []
    for dataset_name, frame in data.items():
        for column, count in frame.isna().sum().items():
            if count:
                missing_rows.append(
                    {
                        "dataset": f"{dataset_name}.csv",
                        "column": column,
                        "missing_rows": int(count),
                        "missing_pct": round(count / len(frame) * 100, 1),
                    }
                )
    missing_table = pd.DataFrame(missing_rows).sort_values(
        ["missing_rows", "dataset", "column"], ascending=[False, True, True]
    )

    frame = add_buckets(feature_table)
    acquisition_rates = grouped_rates(frame, "acquisition_channel").set_index("acquisition_channel")
    recency_rates = grouped_rates(frame, "recency_bucket").set_index("recency_bucket")
    frequency_rates = grouped_rates(frame, "frequency_bucket").set_index("frequency_bucket")
    monetary_rates = grouped_rates(frame, "monetary_bucket").set_index("monetary_bucket")
    return_rates = grouped_rates(frame, "return_rate_bucket").set_index("return_rate_bucket")

    report = f"""# Data Quality Report

## Package Manifest

{package_manifest.to_markdown(index=False)}

## Loaded Dataset Inspection

{schema_summary.to_markdown(index=False)}

## Missing Values

{missing_table.to_markdown(index=False)}

## Duplicate and Duplicate-Like Records

{duplicate_summary.to_markdown(index=False)}

`orders.csv` contains intentionally duplicate-like rows whose `order_id` ends with `_DUP`. Those should be removed or collapsed into their base order before any customer aggregation.

Sample duplicate-like rows:

{duplicate_samples.to_markdown(index=False)}

## Join / Key Issues

{key_health.to_markdown(index=False)}

All customer-linked tables join back cleanly to the 2,400-customer universe; the main integrity risk is duplicate handling rather than orphaned IDs.

## Invalid or Unusual Values

{invalid_checks.to_markdown(index=False)}

## Date Consistency Checks

{date_checks.to_markdown(index=False)}

## Leakage-Sensitive Columns and Rows

{leakage_table.to_markdown(index=False)}

## Outlier Audit

{outlier_summary.to_markdown(index=False)}

Top gross-amount outliers:

{top_order_outliers.to_markdown(index=False)}

## Treatment Recommendations

1. Deduplicate the 12 `_DUP` rows in `orders.csv` before any order-count or spend aggregation.
2. Enforce the snapshot boundary strictly: post-snapshot orders belong to label construction, not model features.
3. Keep true missingness visible for `loyalty_tier`, `skin_type`, and `rating`; encode it rather than silently dropping rows.
4. Winsorize or log-transform `gross_amount` before modelling because spend outliers are extreme enough to dominate averages.
5. Audit `intervention_history.csv` before campaign ROI work because campaign-cost fields are internally inconsistent for hundreds of customers.

## Business-Facing Readout

1. Recency is the strongest warning sign: customers with 121+ day recency churn at **{recency_rates.loc['121+', 'churn_rate_pct']:.1f}%** versus **{recency_rates.loc['0-30', 'churn_rate_pct']:.1f}%** for customers who purchased in the last month.
2. Thin recent order depth matters: customers with only one order in the last 180 days churn at **{frequency_rates.loc['1', 'churn_rate_pct']:.1f}%**, while the 5+ order group drops to **{frequency_rates.loc['5+', 'churn_rate_pct']:.1f}%**.
3. Low spend depth is risky: the bottom spend quartile churns at **{monetary_rates.loc['Q1 Low', 'churn_rate_pct']:.1f}%** versus **{monetary_rates.loc['Q4 High', 'churn_rate_pct']:.1f}%** for the top quartile.
4. Return-heavy customers are fragile: customers with 50%+ return rates churn at **{return_rates.loc['50%+', 'churn_rate_pct']:.1f}%**.
5. Acquisition quality varies: Google Search customers churn at **{acquisition_rates.loc['Google Search', 'churn_rate_pct']:.1f}%**, materially above the **{acquisition_rates.loc['Organic', 'churn_rate_pct']:.1f}%** seen in Organic acquisition.
"""
    return report


def generate_eda_summary(data: dict[str, pd.DataFrame], feature_table: pd.DataFrame) -> str:
    frame = add_buckets(feature_table)
    issue_summary = support_issue_summary(data)
    last_campaign, manual_priority = campaign_summary(data)

    acquisition_table = grouped_rates(frame, "acquisition_channel").sort_values("churn_rate_pct", ascending=False)
    age_table = grouped_rates(frame, "age_group").sort_values("churn_rate_pct", ascending=False)
    city_tier_table = grouped_rates(frame, "city_tier").sort_values("churn_rate_pct", ascending=False)
    consent_table = grouped_rates(frame, "marketing_consent").sort_values("churn_rate_pct", ascending=False)

    recency_table = grouped_rates(frame, "recency_bucket")
    frequency_table = grouped_rates(frame, "frequency_bucket")
    monetary_table = grouped_rates(frame, "monetary_bucket")
    return_table = grouped_rates(frame, "return_rate_bucket")
    sessions_table = grouped_rates(frame, "sessions_bucket")
    diversity_table = grouped_rates(frame, "category_diversity_bucket")

    churn_vs_spend = (
        frame.groupby("churn_next_60d")
        .agg(
            customers=("customer_id", "count"),
            avg_monetary_180d=("monetary_180d", "mean"),
            median_monetary_180d=("monetary_180d", "median"),
            avg_frequency_180d=("frequency_180d", "mean"),
        )
        .reset_index()
        .replace({"churn_next_60d": {0: "Retained", 1: "Churned"}})
        .round(2)
    )

    return f"""## Exploratory Analysis Tables

### Customer Demographics / Profile

**Churn by acquisition channel**

{acquisition_table.to_markdown(index=False)}

**Churn by age group**

{age_table.to_markdown(index=False)}

**Churn by city tier**

{city_tier_table.to_markdown(index=False)}

**Churn by marketing consent**

{consent_table.to_markdown(index=False)}

### Order Behaviour

**Churn by recency**

{recency_table.to_markdown(index=False)}

**Churn by recent order frequency**

{frequency_table.to_markdown(index=False)}

### Monetary Behaviour

**Churn by spend quartile**

{monetary_table.to_markdown(index=False)}

**Spend summary by observed outcome**

{churn_vs_spend.to_markdown(index=False)}

### Support-Ticket Issues

**Customer-level churn by support issue type**

{issue_summary[['issue_type', 'customers_with_issue', 'ticket_count', 'customer_churn_pct', 'avg_resolution_hours', 'avg_sentiment', 'reopened_pct']].round(2).to_markdown(index=False)}

### Return / Refund Behaviour

{return_table.to_markdown(index=False)}

### Web / App Activity

**Churn by 30-day session activity**

{sessions_table.to_markdown(index=False)}

**Churn by category diversity**

{diversity_table.to_markdown(index=False)}

### Campaign / Intervention History

**Churn by most recent campaign**

{last_campaign[['last_campaign_received', 'customers', 'avg_campaign_cost', 'churn_rate_pct']].to_markdown(index=False)}

**Churn by manual CRM priority bucket**

{manual_priority[['manual_priority_bucket', 'customers', 'avg_campaign_cost', 'churn_rate_pct']].to_markdown(index=False)}
"""


def generate_business_memo(data: dict[str, pd.DataFrame], feature_table: pd.DataFrame) -> str:
    frame = add_buckets(feature_table)
    issue_summary = support_issue_summary(data)
    last_campaign, _ = campaign_summary(data)

    recency_rates = grouped_rates(frame, "recency_bucket").set_index("recency_bucket")
    sessions_rates = grouped_rates(frame, "sessions_bucket").set_index("sessions_bucket")
    frequency_rates = grouped_rates(frame, "frequency_bucket").set_index("frequency_bucket")
    monetary_rates = grouped_rates(frame, "monetary_bucket").set_index("monetary_bucket")
    acquisition_rates = grouped_rates(frame, "acquisition_channel").set_index("acquisition_channel")
    return_rates = grouped_rates(frame, "return_rate_bucket").set_index("return_rate_bucket")
    reopened_rates = (
        grouped_rates(frame.loc[frame["ticket_count_90d"] > 0], "reopened_bucket").set_index("reopened_bucket")
    )

    campaign_new_launch = last_campaign.loc[
        last_campaign["last_campaign_received"] == "new_launch", "churn_rate_pct"
    ].iloc[0]
    campaign_none = last_campaign.loc[
        last_campaign["last_campaign_received"] == "none", "churn_rate_pct"
    ].iloc[0]
    most_fragile_issue = issue_summary.iloc[0]["issue_type"]
    most_fragile_issue_churn = issue_summary.iloc[0]["customer_churn_pct"]

    return f"""# Business Memo

## To
Product, CRM, and Customer Support Leaders

## Subject
What the company should investigate before launching a retention campaign

## Executive Summary

Churn is material at **{feature_table['churn_next_60d'].mean() * 100:.1f}%** of the customer base. The main risk pattern is inactivity: customers with 121+ day recency churn at **{recency_rates.loc['121+', 'churn_rate_pct']:.1f}%**, and customers with only 0-2 sessions in the last month churn at **{sessions_rates.loc['0-2', 'churn_rate_pct']:.1f}%**.

## What To Investigate Before Spending Retention Budget

1. **Split dormant customers by recoverability, not just by age since last order.** Customers with 121+ day recency are extremely high risk, but those who still browse occasionally are more recoverable than those with no current activity.
2. **Treat paid-acquisition cohorts differently from Organic or Referral cohorts.** Google Search customers churn at **{acquisition_rates.loc['Google Search', 'churn_rate_pct']:.1f}%** and Instagram customers also sit near 50%, materially above Organic at **{acquisition_rates.loc['Organic', 'churn_rate_pct']:.1f}%**.
3. **Protect recent value before chasing the coldest names.** One-order customers churn at **{frequency_rates.loc['1', 'churn_rate_pct']:.1f}%**, and the lowest spend quartile churns at **{monetary_rates.loc['Q1 Low', 'churn_rate_pct']:.1f}%**. The team should separate low-value churn from high-value churn instead of treating them with the same offer.
4. **Diagnose operational friction before defaulting to discounts.** Customers with 50%+ return rates churn at **{return_rates.loc['50%+', 'churn_rate_pct']:.1f}%**. Among customers who raised tickets, the 50%+ reopened group churns at **{reopened_rates.loc['50%+', 'churn_rate_pct']:.1f}%**. `{most_fragile_issue}` is the highest-churn issue cohort at **{most_fragile_issue_churn:.1f}%**.
5. **Review whether the current campaign mix is matched to risk.** Customers whose latest touch was `new_launch` still churn at **{campaign_new_launch:.1f}%**, only modestly different from the **{campaign_none:.1f}%** seen for customers with no campaign at all. That pattern suggests targeting and offer design need review before scaling spend.

## Recommended Pilot Structure

Run two controlled pilots before a full launch:

1. A **service-recovery lane** for customers with returns, reopened tickets, or severe support issues.
2. A **reactivation lane** for high-value customers showing stale recency and weak digital activity.

Both pilots should include holdouts so the team measures incremental lift rather than raw return rate.
"""


def generate_hypotheses(data: dict[str, pd.DataFrame], feature_table: pd.DataFrame) -> str:
    frame = add_buckets(feature_table)

    recency = grouped_rates(frame, "recency_bucket").set_index("recency_bucket")
    sessions = grouped_rates(frame, "sessions_bucket").set_index("sessions_bucket")
    frequency = grouped_rates(frame, "frequency_bucket").set_index("frequency_bucket")
    monetary = grouped_rates(frame, "monetary_bucket").set_index("monetary_bucket")
    acquisition = grouped_rates(frame, "acquisition_channel").set_index("acquisition_channel")
    returns = grouped_rates(frame, "return_rate_bucket").set_index("return_rate_bucket")
    reopened = grouped_rates(frame.loc[frame["ticket_count_90d"] > 0], "reopened_bucket").set_index(
        "reopened_bucket"
    )

    return f"""## Churn-Risk Hypotheses

1. **Stale recency is the strongest churn signal.** Supported by *Churn Rate by Days Since Last Order*. Customers with 121+ day recency churn at **{recency.loc['121+', 'churn_rate_pct']:.1f}%**, versus **{recency.loc['0-30', 'churn_rate_pct']:.1f}%** for customers who ordered in the last month.
2. **Weak recent web/app activity is an early churn warning.** Supported by *Churn Rate by 30-Day Session Activity*. Customers with only 0-2 sessions churn at **{sessions.loc['0-2', 'churn_rate_pct']:.1f}%**, while the 9+ session group drops to **{sessions.loc['9+', 'churn_rate_pct']:.1f}%**.
3. **Thin recent order frequency signals fragile habit formation.** Supported by *Churn Rate by Order Frequency in the Last 180 Days*. Customers with a single recent order churn at **{frequency.loc['1', 'churn_rate_pct']:.1f}%**, compared with **{frequency.loc['5+', 'churn_rate_pct']:.1f}%** for the 5+ order cohort.
4. **Low spend depth is associated with higher churn.** Supported by *Churn Rate by 180-Day Spend Quartile*. The lowest spend quartile churns at **{monetary.loc['Q1 Low', 'churn_rate_pct']:.1f}%**, versus **{monetary.loc['Q4 High', 'churn_rate_pct']:.1f}%** for the top quartile.
5. **Acquisition quality differs by channel.** Supported by *Churn Rate by Acquisition Channel*. Google Search customers churn at **{acquisition.loc['Google Search', 'churn_rate_pct']:.1f}%** and Instagram customers at **{acquisition.loc['Instagram', 'churn_rate_pct']:.1f}%**, both above Organic at **{acquisition.loc['Organic', 'churn_rate_pct']:.1f}%**.
6. **Returns and unresolved service friction likely suppress repeat purchase intent.** Supported by *Churn Rate by Return Rate in the Last 180 Days* and *Churn Rate by Reopened Ticket Share*. Customers with 50%+ return rates churn at **{returns.loc['50%+', 'churn_rate_pct']:.1f}%**, and ticketed customers with 50%+ reopened cases churn at **{reopened.loc['50%+', 'churn_rate_pct']:.1f}%**.
"""


def build_notebook(
    output_path: Path,
    package_manifest: pd.DataFrame,
    schema_summary: pd.DataFrame,
    join_summary: pd.DataFrame,
    feature_sample: pd.DataFrame,
    quality_report: str,
    eda_summary: str,
    business_memo: str,
    hypotheses: str,
    chart_specs: list[dict[str, str]],
) -> None:
    chart_sections = []
    for spec in chart_specs:
        chart_sections.append(
            f"### {spec['title']}\n![{spec['title']}](charts/{spec['path']})\n\n{spec['caption']}"
        )

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Part 1 — Data Audit, EDA & Business Understanding\n"
            "This notebook is generated by `build_part1.py`. It now documents package inspection, data quality checks, exploratory analysis, churn-risk hypotheses, and the business memo."
        ),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """\
                from build_part1 import find_data_dir, load_data, build_feature_table

                data_dir = find_data_dir()
                data = load_data(data_dir)
                feature_table = build_feature_table(data)

                {name: df.shape for name, df in data.items()}
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Package Inspection\n\n"
            + package_manifest.to_markdown(index=False)
            + "\n\n## Schema Summary\n\n"
            + schema_summary.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(
            "## Join Flow\n\n"
            "The analysis is done at customer level. Raw transactional and interaction tables are aggregated first and then left-joined back to the 2,400-customer base.\n\n"
            + join_summary.to_markdown(index=False)
        ),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """\
                from build_part1 import SNAPSHOT_DATE

                customers = data["customers"].copy()
                orders = data["orders"].copy()
                support = data["support_tickets"].copy()
                web = data["web_events_snapshot"].copy()
                labels = data["churn_labels"].copy()
                interventions = data["intervention_history"].copy()

                pre_orders = orders.loc[orders["order_date"] <= SNAPSHOT_DATE].copy()
                orders_180d = pre_orders.loc[pre_orders["order_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=180)].copy()
                tickets_90d = support.loc[support["ticket_date"] >= SNAPSHOT_DATE - pd.Timedelta(days=90)].copy()

                order_agg = pre_orders.groupby("customer_id").agg(last_order_date=("order_date", "max")).reset_index()
                order_180 = orders_180d.groupby("customer_id").agg(
                    frequency_180d=("order_id", "nunique"),
                    monetary_180d=("gross_amount", "sum"),
                ).reset_index()
                support_90 = tickets_90d.groupby("customer_id").agg(ticket_count_90d=("ticket_id", "count")).reset_index()

                joined = (
                    customers.merge(order_agg, on="customer_id", how="left")
                    .merge(order_180, on="customer_id", how="left")
                    .merge(web, on="customer_id", how="left")
                    .merge(support_90, on="customer_id", how="left")
                    .merge(interventions, on="customer_id", how="left")
                    .merge(labels[["customer_id", "churn_next_60d"]], on="customer_id", how="left")
                )

                joined.shape, joined["customer_id"].nunique()
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "## Joined Customer Feature Sample\n\n"
            "This sample shows the customer-level analysis table after joins and pre-EDA feature preparation.\n\n"
            + feature_sample.to_markdown(index=False)
        ),
        nbf.v4.new_markdown_cell(quality_report),
        nbf.v4.new_markdown_cell("## Visual EDA\n\n" + "\n\n".join(chart_sections)),
        nbf.v4.new_markdown_cell(eda_summary),
        nbf.v4.new_markdown_cell(hypotheses),
        nbf.v4.new_markdown_cell(business_memo),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """\
                # Rebuild the part outputs from the command line:
                # python build_part1.py
                """
            )
        ),
    ]
    nbf.write(notebook, output_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = find_data_dir()
    data = load_data(data_dir)

    package_manifest = build_package_manifest(data_dir, data)
    schema_summary = build_schema_summary(data)
    feature_table = build_feature_table(data)
    join_summary = build_join_summary(data, feature_table)
    feature_sample = feature_table[
        [
            "customer_id",
            "recency_days",
            "frequency_180d",
            "monetary_180d",
            "sessions_30d",
            "ticket_count_90d",
            "last_campaign_received",
            "churn_next_60d",
        ]
    ].sort_values("customer_id").head(12)
    chart_specs = create_charts(data, feature_table, root / "charts")
    quality_report = generate_quality_report(data_dir, data, feature_table)
    eda_summary = generate_eda_summary(data, feature_table)
    hypotheses = generate_hypotheses(data, feature_table)
    business_memo = generate_business_memo(data, feature_table)

    (root / "data_quality_report.md").write_text(quality_report, encoding="utf-8")
    (root / "business_memo.md").write_text(business_memo, encoding="utf-8")
    build_notebook(
        root / "eda_audit.ipynb",
        package_manifest,
        schema_summary,
        join_summary,
        feature_sample,
        quality_report,
        eda_summary,
        business_memo,
        hypotheses,
        chart_specs,
    )
    print(f"Part 1 outputs written to {root}")


if __name__ == "__main__":
    main()
