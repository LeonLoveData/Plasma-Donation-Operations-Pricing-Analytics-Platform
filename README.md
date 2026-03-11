# Plasma Donation Operations & Pricing Analytics Platform

End-to-end data analytics pipeline for **plasma donation operations, pricing analytics, donor behavior modeling, and retention prediction**.

This project simulates how an analytics team at a large **plasma collection network** could use data science to support:

- donor retention strategy
- compensation pricing optimization
- donation volume forecasting
- operational decision making

The platform integrates **data engineering, statistical analysis, machine learning, segmentation, and interactive dashboards** into a reproducible Python workflow.

---

# Project Overview

Plasma collection centers rely on donor participation to maintain a stable supply of plasma used in life-saving therapies.  
Understanding how **compensation levels, donor characteristics, and center operations affect donation behavior** is critical for operational and pricing strategy.

This project builds an analytics system that:

- aggregates donor and center operational data
- analyzes donation patterns and pricing elasticity
- predicts donor churn risk
- segments donors based on behavioral characteristics
- visualizes operational KPIs through an interactive dashboard

The pipeline transforms raw operational data into **actionable insights for plasma center management and pricing strategy**.

---

# Repository Structure
```
plasma-donor-analytics
│
├── data
│ ├── raw
│ │ ├── dim_donor.csv
│ │ ├── dim_center.csv
│ │ ├── dim_date.csv
│ │ ├── fact_experiment.csv
│ │ ├── fact_pricing_config.csv
│ │ └── fact_donation.parquet
│ │
│ └── processed
│ ├── donor_month_panel.parquet
│ ├── center_month_panel.parquet
│ ├── donor_segments.parquet
│ ├── donor_month_with_segments.parquet
│ ├── churn_predictions.parquet
│ └── high_risk_donors.xlsx
│
├── outputs
│ └── figures
│ ├── monthly_total_donations.png
│ ├── center_compensation_vs_donations.png
│ ├── churn_rate_by_region.png
│ └── churn_probability_distribution.png
│
├── models
│ ├── churn_model.joblib
│ ├── donor_kmeans.joblib
│ └── donor_scaler.joblib
│
├── pipeline.py
└── README.md
```
