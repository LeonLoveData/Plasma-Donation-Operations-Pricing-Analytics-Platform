import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

try:
    import streamlit as st
    import plotly.express as px
except ImportError:
    st = None
    px = None

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("outputs/figures")
MODEL_DIR = Path("models")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading raw data...")

    dim_donor = pd.read_csv(RAW_DIR / "dim_donor.csv")
    dim_center = pd.read_csv(RAW_DIR / "dim_center.csv")
    dim_date = pd.read_csv(RAW_DIR / "dim_date.csv")
    fact_experiment = pd.read_csv(RAW_DIR / "fact_experiment.csv")
    fact_pricing_config = pd.read_csv(RAW_DIR / "fact_pricing_config.csv")
    fact_donation = pd.read_parquet(RAW_DIR / "fact_donation.parquet")

    fact_donation["date"] = pd.to_datetime(fact_donation["date"])
    dim_date["date"] = pd.to_datetime(dim_date["date"])
    dim_donor["first_donation_date"] = pd.to_datetime(dim_donor["first_donation_date"])

    return (
        dim_donor,
        dim_center,
        dim_date,
        fact_experiment,
        fact_pricing_config,
        fact_donation,
    )

def build_donor_month_panel(
    dim_donor: pd.DataFrame,
    dim_center: pd.DataFrame,
    fact_donation: pd.DataFrame,
) -> pd.DataFrame:
    print("Building donor_month_panel...")

    df = fact_donation.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    donor_month = (
        df.groupby(["donor_id", "center_id", "year", "month", "year_month"])
        .agg(
            donations_in_month=("donation_id", "count"),
            total_compensation_in_month=("total_compensation", "sum"),
            avg_compensation_per_donation=("total_compensation", "mean"),
            donation_volume_sum=("donation_volume_ml", "sum"),
            last_donation_date=("date", "max"),
        )
        .reset_index()
    )

    donor_month = donor_month.merge(
        dim_donor[
            [
                "donor_id",
                "gender",
                "age",
                "income_bracket",
                "employment_status",
                "signup_channel",
                "distance_to_center_km",
                "first_donation_date",
            ]
        ],
        on="donor_id",
        how="left",
    )

    donor_month = donor_month.merge(
        dim_center[["center_id", "region", "state", "city", "center_type", "capacity_bucket"]],
        on="center_id",
        how="left",
    )

    donor_month = donor_month.sort_values(["donor_id", "year", "month"])
    donor_month["is_last_month_for_donor"] = (
        donor_month.groupby("donor_id")["year_month"].transform("max") == donor_month["year_month"]
    )
    donor_month["is_churned_next_60d"] = donor_month["is_last_month_for_donor"].astype(int)
    donor_month.drop(columns=["is_last_month_for_donor"], inplace=True)

    donor_month.to_parquet(PROCESSED_DIR / "donor_month_panel.parquet", index=False)
    print(f"donor_month_panel saved to {PROCESSED_DIR / 'donor_month_panel.parquet'}")

    return donor_month

def build_center_month_panel(
    dim_center: pd.DataFrame,
    fact_donation: pd.DataFrame,
) -> pd.DataFrame:
    print("Building center_month_panel...")

    df = fact_donation.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    center_month = (
        df.groupby(["center_id", "year", "month", "year_month"])
        .agg(
            total_donations=("donation_id", "count"),
            unique_donors=("donor_id", "nunique"),
            avg_compensation=("total_compensation", "mean"),
            total_compensation=("total_compensation", "sum"),
            total_volume_ml=("donation_volume_ml", "sum"),
        )
        .reset_index()
    )

    center_month = center_month.merge(
        dim_center[["center_id", "region", "state", "city", "center_type", "capacity_bucket"]],
        on="center_id",
        how="left",
    )

    center_month.to_parquet(PROCESSED_DIR / "center_month_panel.parquet", index=False)
    print(f"center_month_panel saved to {PROCESSED_DIR / 'center_month_panel.parquet'}")

    return center_month

def run_data_analysis(
    donor_month: pd.DataFrame,
    center_month: pd.DataFrame,
) -> None:
    print("\n=== Data Analysis ===")

    total_donations = center_month["total_donations"].sum()
    total_unique_donors = donor_month["donor_id"].nunique()
    avg_comp = center_month["avg_compensation"].mean()

    print(f"Total donations: {total_donations:,}")
    print(f"Total unique donors: {total_unique_donors:,}")
    print(f"Average compensation per donation: ${avg_comp:.2f}")

    churn_rate = donor_month["is_churned_next_60d"].mean()
    print(f"Approx. churn rate (donor-month last-month proxy): {churn_rate:.2%}")

    churn_by_region = (
        donor_month.groupby("region")["is_churned_next_60d"].mean().sort_values(ascending=False)
    )
    print("\nChurn rate by region:")
    print(churn_by_region.to_frame("churn_rate"))

    top_centers = (
        center_month.groupby(["center_id", "region", "city"])["total_donations"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 centers by total donations:")
    print(top_centers)

    cm = center_month.copy()
    cm = cm[(cm["total_donations"] > 0) & (cm["avg_compensation"] > 0)]

    cm["log_donations"] = np.log(cm["total_donations"])
    cm["log_avg_comp"] = np.log(cm["avg_compensation"])

    X = cm[["log_avg_comp"]].values
    y = cm["log_donations"].values

    model = LinearRegression()
    model.fit(X, y)
    elasticity = model.coef_[0]

    print(f"\nEstimated price elasticity (log-log, center-month level): {elasticity:.3f}")
    print("Interpretation: 1% increase in compensation is associated with "
          f"{elasticity:.2%} change in donations (approx).")

def create_visualizations(
    donor_month: pd.DataFrame,
    center_month: pd.DataFrame,
) -> None:
    print("\nCreating visualizations...")

    sns.set(style="whitegrid")

    monthly_trend = (
        center_month.groupby("year_month")["total_donations"].sum().reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_trend, x="year_month", y="total_donations", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Total Donations Over Time")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Donations")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_total_donations.png", dpi=150)
    plt.close()

    center_agg = (
        center_month.groupby(["center_id", "region"])
        .agg(
            avg_compensation=("avg_compensation", "mean"),
            total_donations=("total_donations", "sum"),
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=center_agg,
        x="avg_compensation",
        y="total_donations",
        hue="region",
        palette="tab10",
    )
    plt.title("Center-level: Avg Compensation vs Total Donations")
    plt.xlabel("Average Compensation ($)")
    plt.ylabel("Total Donations")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "center_compensation_vs_donations.png", dpi=150)
    plt.close()

    churn_by_region = (
        donor_month.groupby("region")["is_churned_next_60d"].mean().reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=churn_by_region.sort_values("is_churned_next_60d", ascending=False),
        x="region",
        y="is_churned_next_60d",
        palette="Reds",
    )
    plt.title("Churn Rate by Region (Donor-Month Last-Month Proxy)")
    plt.xlabel("Region")
    plt.ylabel("Churn Rate")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_rate_by_region.png", dpi=150)
    plt.close()

    print(f"Figures saved to: {FIG_DIR}")

def train_churn_model(donor_month: pd.DataFrame) -> None:
    print("\nTraining churn model...")

    df = donor_month.copy()

    feature_cols = [
        "donations_in_month",
        "total_compensation_in_month",
        "avg_compensation_per_donation",
        "donation_volume_sum",
        "age",
        "distance_to_center_km",
    ]
    df = df.dropna(subset=feature_cols + ["is_churned_next_60d"])

    X = df[feature_cols].values
    y = df["is_churned_next_60d"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Churn model AUC: {auc:.3f}")

    joblib.dump(model, MODEL_DIR / "churn_model.joblib")
    print(f"Churn model saved to {MODEL_DIR / 'churn_model.joblib'}")

def run_segmentation(donor_month: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning donor segmentation...")

    donor_agg = (
        donor_month.groupby("donor_id")
        .agg(
            avg_donations_in_month=("donations_in_month", "mean"),
            avg_total_compensation=("total_compensation_in_month", "mean"),
            avg_donation_volume=("donation_volume_sum", "mean"),
            age=("age", "mean"),
            distance_to_center_km=("distance_to_center_km", "mean"),
        )
        .reset_index()
    )

    feature_cols = [
        "avg_donations_in_month",
        "avg_total_compensation",
        "avg_donation_volume",
        "age",
        "distance_to_center_km",
    ]

    X = donor_agg[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    donor_agg["segment"] = kmeans.fit_predict(X_scaled)

    donor_agg.to_parquet(PROCESSED_DIR / "donor_segments.parquet", index=False)
    joblib.dump(kmeans, MODEL_DIR / "donor_kmeans.joblib")
    joblib.dump(scaler, MODEL_DIR / "donor_scaler.joblib")

    print(f"Donor segments saved to {PROCESSED_DIR / 'donor_segments.parquet'}")

    donor_month_with_seg = donor_month.merge(
        donor_agg[["donor_id", "segment"]], on="donor_id", how="left"
    )
    donor_month_with_seg.to_parquet(PROCESSED_DIR / "donor_month_with_segments.parquet", index=False)
    print(f"donor_month_with_segments saved to {PROCESSED_DIR / 'donor_month_with_segments.parquet'}")

    return donor_month_with_seg

def run_dashboard():
    if st is None or px is None:
        raise ImportError("streamlit 和 plotly 未安装，请先 `pip install streamlit plotly`")

    st.set_page_config(page_title="ABC_Company Pricing & Donor Analytics", layout="wide")

    st.title("ABC_Company Plasma Pricing & Donor Analytics Dashboard")

    donor_month_path = PROCESSED_DIR / "donor_month_with_segments.parquet"
    if donor_month_path.exists():
        donor_month = pd.read_parquet(donor_month_path)
    else:
        donor_month = pd.read_parquet(PROCESSED_DIR / "donor_month_panel.parquet")

    center_month = pd.read_parquet(PROCESSED_DIR / "center_month_panel.parquet")

    regions = ["All"] + sorted(center_month["region"].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)

    centers = ["All"]
    if selected_region != "All":
        centers += (
            center_month[center_month["region"] == selected_region]["center_id"]
            .dropna()
            .unique()
            .tolist()
        )
    else:
        centers += center_month["center_id"].dropna().unique().tolist()

    selected_center = st.sidebar.selectbox("Center", centers)

    cm = center_month.copy()
    dm = donor_month.copy()

    if selected_region != "All":
        cm = cm[cm["region"] == selected_region]
        dm = dm[dm["region"] == selected_region]

    if selected_center != "All":
        cm = cm[cm["center_id"] == selected_center]
        dm = dm[dm["center_id"] == selected_center]

    total_donations = cm["total_donations"].sum()
    total_unique_donors = dm["donor_id"].nunique()
    churn_rate = dm["is_churned_next_60d"].mean() if "is_churned_next_60d" in dm.columns else np.nan
    avg_comp = cm["avg_compensation"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Donations", f"{int(total_donations):,}")
    col2.metric("Unique Donors", f"{int(total_unique_donors):,}")
    col3.metric("Churn Rate", f"{churn_rate:.1%}" if not np.isnan(churn_rate) else "N/A")
    col4.metric("Avg Compensation", f"${avg_comp:.2f}" if not np.isnan
