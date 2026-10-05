import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Financial Fraud Analytics & ML Predictor",
    page_icon="🛡️",
    layout="wide"
)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "banking_fraud.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_pipeline.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

@st.cache_resource
def load_ml_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def fetch_recent_transactions():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT t.transaction_id, t.amount, t.transaction_date, t.account_id,
           s.risk_score, s.status, s.is_fraud, l.city
    FROM Transaction_record t
    LEFT JOIN Suspicious_transaction s ON t.transaction_id = s.transaction_id
    LEFT JOIN Transaction_location l ON t.transaction_id = l.transaction_id
    ORDER BY t.transaction_date DESC
    LIMIT 50
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Main Title & Header
st.title("🛡️ Financial Fraud Analytics & Machine Learning System")
st.markdown("---")

pipeline_data = load_ml_pipeline()

# Navigation Sidebar
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Module:",
    ["🚨 Real-Time Fraud Predictor", "📊 Visual Data Analytics & EDA", "🔍 Investigation Queue", "⚡ Model Performance Metrics"]
)

# ---------------------------------------------------------
# PAGE 1: REAL-TIME FRAUD PREDICTOR
# ---------------------------------------------------------
if page == "🚨 Real-Time Fraud Predictor":
    st.header("🚨 Real-Time Transaction Fraud Risk Predictor")
    st.write("Input transaction details to assess immediate fraud probability and risk score using ML.")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount (INR)", min_value=1.0, max_value=1000000.0, value=75000.0, step=1000.0)
        balance = st.number_input("Account Balance (INR)", min_value=0.0, max_value=5000000.0, value=120000.0, step=5000.0)
        account_type = st.selectbox("Account Type", ["Savings", "Checking", "Fixed Deposit", "Salary", "Recurring Deposit"])

    with col2:
        tx_type_id = st.selectbox("Transaction Type", [
            (3, "NEFT Transfer"), (4, "RTGS Transfer"), (5, "IMPS Transfer"),
            (6, "UPI Payment"), (7, "ATM Cash Withdrawal"), (16, "POS Purchase"), (17, "Online Purchase")
        ], format_func=lambda x: f"{x[1]} (ID {x[0]})")[0]
        
        hour_of_day = st.slider("Hour of Day (0 - 23)", min_value=0, max_value=23, value=2)
        day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

    with col3:
        tx_count_10m = st.number_input("Txns in Past 10 Mins", min_value=0, max_value=20, value=6)
        tx_count_1h = st.number_input("Txns in Past 1 Hour", min_value=0, max_value=50, value=12)
        travel_speed_kmh = st.number_input("Geographic Travel Speed (km/h)", min_value=0.0, max_value=3000.0, value=950.0)

    st.markdown("---")
    
    if st.button("🔍 Assess Fraud Risk", type="primary"):
        if pipeline_data is None:
            st.error("Model pipeline not found. Please train models first.")
        else:
            model = pipeline_data['model']
            preprocessor = pipeline_data['preprocessor']
            feature_cols = pipeline_data['feature_cols']

            # Derived Features
            is_night_transaction = 1 if 1 <= hour_of_day <= 4 else 0
            tx_count_24h = tx_count_1h * 2
            tx_sum_1h = amount * (tx_count_1h / 2.0)
            amount_to_avg_ratio = amount / 2500.0
            amount_to_balance_ratio = amount / (balance + 1e-5)
            impossible_speed_flag = 1 if travel_speed_kmh > 800 else 0

            # Construct Input DataFrame
            input_dict = {
                'amount': [amount],
                'balance': [balance],
                'account_type': [account_type],
                'transaction_type_id': [tx_type_id],
                'hour_of_day': [hour_of_day],
                'day_of_week': [day_of_week],
                'is_night_transaction': [is_night_transaction],
                'tx_count_10m': [tx_count_10m],
                'tx_count_1h': [tx_count_1h],
                'tx_count_24h': [tx_count_24h],
                'tx_sum_1h': [tx_sum_1h],
                'travel_speed_kmh': [travel_speed_kmh],
                'amount_to_avg_ratio': [amount_to_avg_ratio],
                'amount_to_balance_ratio': [amount_to_balance_ratio],
                'impossible_speed_flag': [impossible_speed_flag]
            }

            input_df = pd.DataFrame(input_dict)[feature_cols]

            # Preprocess and Predict
            X_proc = preprocessor.transform(input_df)
            fraud_prob = model.predict_proba(X_proc)[0, 1] if hasattr(model, "predict_proba") else (0.9 if model.predict(X_proc)[0] == 1 else 0.1)
            risk_score = int(fraud_prob * 100)

            # Result Display Cards
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(label="Fraud Probability", value=f"{fraud_prob*100:.1f}%")
            with res_col2:
                st.metric(label="Risk Score", value=f"{risk_score} / 100")
            with res_col3:
                if risk_score >= 75:
                    st.error("🚨 CRITICAL FRAUD RISK")
                elif risk_score >= 45:
                    st.warning("⚠️ MEDIUM SUSPICION RISK")
                else:
                    st.success("✅ LOW RISK / LEGITIMATE")

            # Risk Factor Explanations
            st.subheader("🚩 Risk Factors Identified:")
            risk_factors = []
            if impossible_speed_flag == 1:
                risk_factors.append("🌐 **Impossible Geographic Speed**: Travel speed exceeds 800 km/h between locations.")
            if tx_count_10m > 4:
                risk_factors.append(f"⚡ **High Velocity Burst**: {tx_count_10m} transactions within 10 minutes.")
            if is_night_transaction == 1 and amount > 50000:
                risk_factors.append("🌙 **Off-Hour High Value Transfer**: Transaction occurred late night (1 AM - 4 AM).")
            if amount_to_balance_ratio > 0.8:
                risk_factors.append("💰 **High Balance Depletion Ratio**: Transaction depletes over 80% of account balance.")

            if risk_factors:
                for rf in risk_factors:
                    st.write(rf)
            else:
                st.write("🟢 No abnormal transactional behaviors detected.")

# ---------------------------------------------------------
# PAGE 2: VISUAL DATA ANALYTICS & EDA
# ---------------------------------------------------------
elif page == "📊 Visual Data Analytics & EDA":
    st.header("📊 Exploratory Data Analysis & Visual Insights")
    st.write("Visualizations generated from database feature matrices.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Amount Distribution")
        dist_path = os.path.join(REPORTS_DIR, "transaction_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, use_container_width=True)

        st.subheader("Geographic Fraud Mapping")
        geo_path = os.path.join(REPORTS_DIR, "geographic_fraud_map.png")
        if os.path.exists(geo_path):
            st.image(geo_path, use_container_width=True)

    with col2:
        st.subheader("Hourly Fraud Rate Spikes")
        hourly_path = os.path.join(REPORTS_DIR, "hourly_fraud_rate.png")
        if os.path.exists(hourly_path):
            st.image(hourly_path, use_container_width=True)

        st.subheader("Feature Correlation Heatmap")
        corr_path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
        if os.path.exists(corr_path):
            st.image(corr_path, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: INVESTIGATION QUEUE
# ---------------------------------------------------------
elif page == "🔍 Investigation Queue":
    st.header("🔍 Suspicious Transaction Monitoring Queue")
    st.write("Live database feed of flagged transactions requiring investigator review.")

    recent_df = fetch_recent_transactions()
    if not recent_df.empty:
        st.dataframe(
            recent_df,
            column_config={
                "amount": st.column_config.NumberColumn("Amount (INR)", format="₹%.2f"),
                "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d"),
                "is_fraud": st.column_config.CheckboxColumn("Confirmed Fraud?")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No transaction data available in local database.")

# ---------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE METRICS
# ---------------------------------------------------------
elif page == "⚡ Model Performance Metrics":
    st.header("⚡ Model Performance & Benchmark Evaluation")
    
    if pipeline_data:
        st.subheader(f"🏆 Best Trained Model: **{pipeline_data['model_name']}**")
        metrics = pipeline_data['metrics']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")
        col2.metric("PR-AUC Score", f"{metrics['pr_auc']:.4f}")
        col3.metric("Fraud Recall", f"{metrics['recall']*100:.1f}%")
        col4.metric("Fraud Precision", f"{metrics['precision']*100:.1f}%")

        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = metrics['cm']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
    else:
        st.error("No saved model metrics found.")
