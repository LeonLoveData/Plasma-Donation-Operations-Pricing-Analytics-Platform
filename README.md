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

---

# Data Model

The project uses a **star schema design** commonly used in analytics systems.

### Dimension Tables

- `dim_donor` – donor demographic attributes
- `dim_center` – plasma center characteristics
- `dim_date` – calendar attributes

### Fact Tables

- `fact_donation` – individual plasma donation records
- `fact_pricing_config` – compensation policy configuration
- `fact_experiment` – operational experiment metadata

These tables allow analysis at **donor, center, and time levels**.

---

# Analytical Pipeline

The system executes several analytical modules.

---

# 1 Data Engineering

Raw donation records are transformed into two analytical tables.

### donor_month_panel

Donor-level monthly panel including:

- donation frequency
- compensation received
- donation volume
- donor demographics
- distance to center
- churn indicator

### center_month_panel

Center-level monthly aggregation including:

- total donations
- number of unique donors
- average compensation
- total donation volume

These tables provide the foundation for modeling and operational analytics.

---

# 2 Operational KPI Analysis

The pipeline calculates key operational metrics:

- total donation volume
- number of unique donors
- average compensation per donation
- donor churn rate
- top performing centers

Example insights include:

- donation activity by region
- operational performance differences between centers
- donor retention patterns

---

# 3 Pricing Elasticity Analysis

A log-log regression model estimates the **relationship between donor compensation and donation volume**.


log(total_donations) ~ log(avg_compensation)


The coefficient approximates **donation supply elasticity with respect to compensation**.

This analysis helps evaluate:

- how donor participation responds to compensation changes
- potential pricing strategies to maintain plasma supply

---

# 4 Data Visualization

The pipeline automatically generates operational charts.

### Monthly Donation Trends

Shows how donation activity evolves over time.

### Compensation vs Donation Volume

Center-level relationship between donor compensation and donation volume.

### Churn Rate by Region

Highlights geographic differences in donor retention.

All visualizations are saved to:


outputs/figures


---

# 5 Donor Churn Prediction

A supervised machine learning model predicts **donor churn risk**.

### Model

Gradient Boosting Classifier

### Features

- donations_in_month
- total_compensation_in_month
- avg_compensation_per_donation
- donation_volume_sum
- age
- distance_to_center_km

### Outputs

- predicted churn probability
- model performance (AUC)
- identification of high-risk donors

High-risk donors (top 5%) are automatically exported to support **potential retention outreach strategies**.

Generated files:


churn_predictions.parquet
high_risk_donors.xlsx


---

# 6 Donor Segmentation

Donors are segmented using **K-Means clustering** based on behavioral characteristics.

### Segmentation Features

- donation frequency
- compensation levels
- donation volume
- donor age
- distance to center

The algorithm groups donors into **four behavioral segments**, enabling:

- targeted incentive programs
- personalized donor engagement
- operational planning

Segment results are saved to:


donor_segments.parquet


---

# 7 Interactive Analytics Dashboard

A Streamlit dashboard enables interactive exploration of the analytics results.

Features include:

- operational KPIs
- donation trends over time
- compensation vs donation analysis
- donor segmentation insights

Users can filter by:

- region
- plasma center

This dashboard simulates how business teams can interact with **operational analytics and decision-support data**.

---

# Example Use Cases

This analytics platform demonstrates workflows relevant to healthcare operations analytics:

- donor retention analysis
- compensation strategy evaluation
- donor behavioral segmentation
- operational performance monitoring
- data-driven decision support

---

# Technology Stack

Python

Core libraries:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- streamlit
- plotly

Machine learning methods:

- Gradient Boosting (churn prediction)
- K-Means clustering (segmentation)
- Linear regression (price elasticity)

---

# Project Purpose

This project demonstrates how **data science and analytics can support plasma donation operations**, transforming raw operational data into insights that inform:

- donor engagement strategies
- compensation policy decisions
- operational optimization
- business intelligence reporting
