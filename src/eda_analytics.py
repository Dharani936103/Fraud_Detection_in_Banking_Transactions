import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from feature_engineering import extract_features

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_eda_reports():
    """Generates and saves exploratory data analysis plots for fraud analytics."""
    print("Extracting feature dataset for EDA...")
    df, features = extract_features()

    sns.set_theme(style="whitegrid")

    # 1. Transaction Amount Distribution (Normal vs Fraud)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='amount', hue='is_fraud', kde=True, bins=40, palette={0: 'navy', 1: 'crimson'}, log_scale=True)
    plt.title("Transaction Amount Distribution (Log Scale): Normal vs Fraud", fontsize=14, fontweight='bold')
    plt.xlabel("Amount (INR Log Scale)")
    plt.ylabel("Transaction Count")
    plt.legend(title="Status", labels=["Fraud", "Normal"])
    plt.tight_layout()
    amount_plot = os.path.join(REPORTS_DIR, "transaction_distribution.png")
    plt.savefig(amount_plot, dpi=300)
    plt.close()
    print(f"Saved: {amount_plot}")

    # 2. Hourly Fraud Rate & Velocity Spikes
    plt.figure(figsize=(10, 5))
    hourly_df = df.groupby('hour_of_day')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    sns.barplot(data=hourly_df, x='hour_of_day', y='mean', color='crimson')
    plt.title("Fraud Probability by Hour of Day (Late Night Spikes)", fontsize=14, fontweight='bold')
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    hourly_plot = os.path.join(REPORTS_DIR, "hourly_fraud_rate.png")
    plt.savefig(hourly_plot, dpi=300)
    plt.close()
    print(f"Saved: {hourly_plot}")

    # 3. Geographic Fraud Location Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='longitude', y='latitude', hue='is_fraud', size='amount',
                    sizes=(20, 200), palette={0: 'dodgerblue', 1: 'red'}, alpha=0.7)
    plt.title("Geographic Transaction Distribution & Fraud Clusters", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Is Fraud")
    plt.tight_layout()
    geo_plot = os.path.join(REPORTS_DIR, "geographic_fraud_map.png")
    plt.savefig(geo_plot, dpi=300)
    plt.close()
    print(f"Saved: {geo_plot}")

    # 4. Feature Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = ['amount', 'balance', 'hour_of_day', 'is_night_transaction',
                    'tx_count_10m', 'tx_count_1h', 'travel_speed_kmh',
                    'amount_to_avg_ratio', 'impossible_speed_flag', 'is_fraud']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Engineered Feature Correlation Matrix with Fraud Target", fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_plot = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_plot, dpi=300)
    plt.close()
    print(f"Saved: {heatmap_plot}")

    print("All EDA Reports generated successfully!")

if __name__ == "__main__":
    generate_eda_reports()
