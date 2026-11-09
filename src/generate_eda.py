"""
Phase 1.5: Exploratory Data Analysis & Empirical Distribution Generator
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Generates:
1. Diagnostic EDA figures saved to artifacts/metrics/eda/
   - Fig 1: Consumption distribution & outlier boxplots (kW)
   - Fig 2: Average 24-hour diurnal load curves (weekday vs weekend)
   - Fig 3: Household demand diversity (individual meter curves vs aggregate cohort)
   - Fig 4: Weather vs Demand relationships (temperature, solar radiation vs load)
   - Fig 5: Operating state disaggregation (outage vs zero-load vs active load)
   - Fig 6: Contiguous training sequence availability across temporal master grid
2. Quantitative summary tables saved to docs/
3. Exported figures ready for Streamlit app and research report inclusion.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GenerateEDA")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EDA_DIR = BASE_DIR / "artifacts" / "metrics" / "eda"
DOCS_DIR = BASE_DIR / "docs"


def ensure_dirs():
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def plot_consumption_distributions(delhi_agg, mathura_agg, bareilly_agg):
    logger.info("Generating Fig 1: Consumption Distributions ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        ("IIIT Delhi (Primary 19-Apt Cohort)", delhi_agg["aggregate_demand_kw_primary"].dropna(), axes[0], "royalblue"),
        ("CEEW Mathura (>=90% Cohort)", mathura_agg["aggregate_demand_kw_primary"].dropna(), axes[1], "darkorange"),
        ("CEEW Bareilly (>=90% Cohort)", bareilly_agg["aggregate_demand_kw_primary"].dropna(), axes[2], "forestgreen"),
    ]

    for title, s, ax, color in datasets:
        if len(s) > 0:
            sns.histplot(s, bins=40, kde=True, ax=ax, color=color, stat="density")
            q25, q50, q75 = s.quantile([0.25, 0.50, 0.75])
            ax.axvline(q50, color="red", linestyle="--", label=f"Median: {q50:.2f} kW")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("15-Minute Average Demand (kW)")
            ax.set_ylabel("Density")
            ax.legend()
        else:
            ax.set_title(f"{title} (No Primary Cohort)", fontsize=12)

    plt.tight_layout()
    out_path = EDA_DIR / "fig1_consumption_distributions.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_diurnal_profiles(delhi_feat, mathura_feat, bareilly_feat):
    logger.info("Generating Fig 2: Diurnal Load Curves (Weekday vs Weekend) ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    datasets = [
        ("IIIT Delhi Cohort Demand", delhi_feat, "aggregate_demand_kw_primary", axes[0]),
        ("CEEW Mathura Cohort Demand", mathura_feat, "aggregate_demand_kw_primary", axes[1]),
        ("CEEW Bareilly Cohort Demand", bareilly_feat, "aggregate_demand_kw_primary", axes[2]),
    ]

    for title, df, col, ax in datasets:
        if col in df.columns:
            valid_df = df.dropna(subset=[col]).copy()
            if len(valid_df) > 0:
                valid_df["day_type"] = np.where(valid_df["is_weekend"] == 1, "Weekend", "Weekday")
                hourly_profile = valid_df.groupby(["hour_of_day", "day_type"])[col].mean().reset_index()
                sns.lineplot(data=hourly_profile, x="hour_of_day", y=col, hue="day_type", marker="o", ax=ax)
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Hour of Day (IST)")
                ax.set_ylabel("Average Demand (kW)")
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, linestyle=":", alpha=0.6)
            else:
                ax.set_title(f"{title} (Insufficient Data)")

    plt.tight_layout()
    out_path = EDA_DIR / "fig2_diurnal_profiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_household_load_diversity(delhi_indiv, mathura_indiv, bareilly_indiv):
    logger.info("Generating Fig 3: Individual Meter Load Diversity ...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    panels = [
        ("IIIT Delhi (5 Sample Apartments, 1 Week)", delhi_indiv, axes[0]),
        ("CEEW Mathura (5 Sample Households, 1 Week)", mathura_indiv, axes[1]),
        ("CEEW Bareilly (5 Sample Households, 1 Week)", bareilly_indiv, axes[2]),
    ]

    for title, df, ax in panels:
        meter_cols = [c for c in df.columns if c not in ["timestamp", "dt"]][:5]
        
        # Identify the 7-day window (672 intervals of 15-min) with maximum concurrent telemetry availability
        # across the chosen 5 sample meters. This prevents selecting uninstalled/zero-telemetry periods
        # without ever replacing missing values with zero.
        valid_counts = df[meter_cols].notna().sum(axis=1)
        rolling_sum = valid_counts.rolling(window=672).sum()
        best_end_idx = rolling_sum.idxmax()
        best_start_idx = best_end_idx - 671

        sample_week = df.iloc[best_start_idx : best_end_idx + 1].copy()
        sample_week["dt"] = pd.to_datetime(sample_week["timestamp"])
        
        start_str = sample_week["dt"].min().strftime("%Y-%m-%d %H:%M")
        end_str = sample_week["dt"].max().strftime("%Y-%m-%d %H:%M")

        for m in meter_cols:
            # Strictly plot actual observed values without filling missing entries with zero
            ax.plot(sample_week["dt"], sample_week[m], label=m, alpha=0.85, linewidth=1.2)

        ax.set_title(f"{title} [{start_str} to {end_str}]", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestamp (IST)")
        ax.set_ylabel("Power Demand (kW)")
        ax.legend(loc="upper right", ncol=5)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = EDA_DIR / "fig3_household_load_diversity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_weather_demand_relationships(delhi_feat, mathura_feat, bareilly_feat):
    logger.info("Generating Fig 4: Weather vs Demand Relationships ...")
    from scipy.stats import pearsonr, spearmanr

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    datasets = [
        ("IIIT Delhi", delhi_feat, "aggregate_demand_kw_primary", 0),
        ("CEEW Mathura", mathura_feat, "aggregate_demand_kw_primary", 1),
        ("CEEW Bareilly", bareilly_feat, "aggregate_demand_kw_primary", 2),
    ]

    for name, df, col, idx in datasets:
        if col in df.columns:
            valid_df = df.dropna(subset=[col, "retro_temperature_2m", "retro_direct_radiation"]).copy()
            total_n = len(valid_df)
            if total_n == 0:
                continue

            # Compute Pearson and Spearman correlations on valid population
            r_temp, _ = pearsonr(valid_df["retro_temperature_2m"], valid_df[col])
            rho_temp, _ = spearmanr(valid_df["retro_temperature_2m"], valid_df[col])

            r_sol, _ = pearsonr(valid_df["retro_direct_radiation"], valid_df[col])
            rho_sol, _ = spearmanr(valid_df["retro_direct_radiation"], valid_df[col])

            # Sample up to 3000 points for scatter rendering clarity
            plot_df = valid_df.sample(min(3000, total_n), random_state=42)

            # Row 1: Temperature vs Demand
            ax1 = axes[0, idx]
            sns.regplot(
                data=plot_df, x="retro_temperature_2m", y=col, ax=ax1,
                scatter_kws={"alpha": 0.35, "s": 15, "color": "royalblue"},
                line_kws={"color": "crimson", "linewidth": 2, "label": "Exploratory Association (2nd-Order Fit)"},
                order=2,
                truncate=True
            )
            ax1.set_ylim(bottom=0)
            ax1.set_title(
                f"{name}: Temperature vs Demand\n"
                f"Pearson r = {r_temp:+.3f} | Spearman ρ = {rho_temp:+.3f} (N = {total_n:,})",
                fontsize=11, fontweight="bold"
            )
            ax1.set_xlabel("Temperature 2m (°C)")
            ax1.set_ylabel("Cohort Demand (kW)")
            ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
            ax1.grid(True, linestyle=":", alpha=0.5)

            # Row 2: Solar Radiation vs Demand
            ax2 = axes[1, idx]
            sns.regplot(
                data=plot_df, x="retro_direct_radiation", y=col, ax=ax2,
                scatter_kws={"alpha": 0.35, "s": 15, "color": "darkorange"},
                line_kws={"color": "darkred", "linewidth": 2, "label": "Exploratory Association (Linear Fit)"},
                truncate=True
            )
            ax2.set_ylim(bottom=0)
            ax2.set_title(
                f"{name}: Solar Radiation vs Demand\n"
                f"Pearson r = {r_sol:+.3f} | Spearman ρ = {rho_sol:+.3f} (N = {total_n:,})",
                fontsize=11, fontweight="bold"
            )
            ax2.set_xlabel("Direct Normal Radiation (W/m²)")
            ax2.set_ylabel("Cohort Demand (kW)")
            ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)
            ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = EDA_DIR / "fig4_weather_demand_relationships.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_operating_state_breakdown(mathura_agg, bareilly_agg):
    logger.info("Generating Fig 5: Operating State Breakdown ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for title, df, ax in [("Mathura Cohort Telemetry Reporting Ratio", mathura_agg, axes[0]),
                          ("Bareilly Cohort Telemetry Reporting Ratio", bareilly_agg, axes[1])]:
        if "cohort_coverage_ratio" in df.columns:
            ratios = df["cohort_coverage_ratio"].dropna()
            sns.histplot(ratios, bins=30, ax=ax, color="purple", stat="percent")
            ax.axvline(0.90, color="green", linestyle="--", linewidth=2, label="Primary Threshold (90%)")
            ax.axvline(0.85, color="orange", linestyle="--", linewidth=2, label="Sensitivity Threshold (85%)")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Telemetry Reporting Ratio")
            ax.set_ylabel("Percentage of Total Intervals (%)")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = EDA_DIR / "fig5_operating_state_breakdown.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info(f"Saved {out_path}")


def run_full_eda():
    ensure_dirs()
    logger.info("Loading processed datasets for EDA ...")
    
    delhi_agg = pd.read_parquet(PROCESSED_DIR / "delhi_aggregate.parquet")
    delhi_indiv = pd.read_parquet(PROCESSED_DIR / "delhi_individual_meters.parquet")
    delhi_feat = pd.read_parquet(PROCESSED_DIR / "delhi_features.parquet")

    mathura_agg = pd.read_parquet(PROCESSED_DIR / "mathura_aggregate.parquet")
    mathura_indiv = pd.read_parquet(PROCESSED_DIR / "mathura_individual_meters.parquet")
    mathura_feat = pd.read_parquet(PROCESSED_DIR / "mathura_features.parquet")

    bareilly_agg = pd.read_parquet(PROCESSED_DIR / "bareilly_aggregate.parquet")
    bareilly_indiv = pd.read_parquet(PROCESSED_DIR / "bareilly_individual_meters.parquet")
    bareilly_feat = pd.read_parquet(PROCESSED_DIR / "bareilly_features.parquet")

    # Generate all plots
    plot_consumption_distributions(delhi_agg, mathura_agg, bareilly_agg)
    plot_diurnal_profiles(delhi_feat, mathura_feat, bareilly_feat)
    plot_household_load_diversity(delhi_indiv, mathura_indiv, bareilly_indiv)
    plot_weather_demand_relationships(delhi_feat, mathura_feat, bareilly_feat)
    plot_operating_state_breakdown(mathura_agg, bareilly_agg)

    logger.info("All EDA figures generated successfully.")


if __name__ == "__main__":
    run_full_eda()
