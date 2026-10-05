# 🛡️ Financial Fraud Detection & Analytics System (Data Science & ML Project)

## 📌 Project Overview
This project upgrades a relational database (SQL) banking schema into a **full end-to-end Data Science and Machine Learning platform** for real-time financial fraud detection and behavioral risk scoring.

By extracting transactional data, spatial coordinates, device logs, and customer account details, the system constructs engineered ML features (transaction velocity, geographic speed, amount ratios, temporal flags) and trains supervised and unsupervised machine learning algorithms to detect fraudulent activities.

---

## 🏗️ Architecture & Modules

```text
fraud_detection_ds/
├── data/
│   └── banking_fraud.db           # SQLite database storing relational schemas & 3,000+ transactions
├── src/
│   ├── db_setup.py                # Database initialiser & synthetic transaction generator with fraud patterns
│   ├── feature_engineering.py     # Velocity, spatial distance (Haversine speed), temporal & ratio feature extraction
│   ├── eda_analytics.py           # Exploratory Data Analysis & visual plot generator
│   └── train_models.py            # Model training (SMOTE, Logistic Regression, Random Forest, XGBoost, Isolation Forest)
├── reports/
│   └── figures/                   # Saved EDA visual plots (Heatmaps, Distributions, Geo-maps)
├── models/
│   └── fraud_detection_pipeline.joblib  # Serialised model preprocessor & trained pipeline
├── app.py                         # Interactive Streamlit Web Application (Predictor & Monitoring Dashboard)
└── README.md
```

---

## ⚡ Key Machine Learning Features Engineered

1. **Transaction Velocity Features**:
   - `tx_count_10m`: Number of transactions by customer in past 10 minutes.
   - `tx_count_1h`: Number of transactions in past 1 hour.
   - `tx_count_24h`: Number of transactions in past 24 hours.
   - `tx_sum_1h`: Total money transferred in past 1 hour.

2. **Spatial Velocity (Geographic Distance Speed)**:
   - Uses the **Haversine Formula** to compute distance ($\text{km}$) between consecutive login/transaction coordinates ($\text{Lat}, \text{Lon}$).
   - `travel_speed_kmh`: Calculated speed in $\text{km/h} = \frac{\Delta \text{Distance}}{\Delta \text{Time}}$.
   - `impossible_speed_flag`: Flags physical movement exceeding $800\text{ km/h}$.

3. **Amount Ratios & Relative Features**:
   - `amount_to_avg_ratio`: Ratio of current transaction amount to customer's 30-day historical mean.
   - `amount_to_balance_ratio`: Ratio of transaction amount to total available account balance.

4. **Temporal & Off-Hour Features**:
   - `hour_of_day`: Transaction hour ($0 - 23$).
   - `is_night_transaction`: Binary flag for high-risk late-night hours ($1\text{ AM} - 4\text{ AM}$).

---

## 🏆 Model Evaluation Metrics & Benchmarks

To handle severe class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set.

| Algorithm | ROC-AUC Score | PR-AUC Score | Fraud Recall | Fraud Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **0.9853** | **0.8732** | **92.86%** | **59.09%** | **0.7222** |
| **Random Forest Classifier** | 0.9906 | 0.8660 | 82.14% | 62.16% | 0.7077 |
| **XGBoost Classifier** | 0.9889 | 0.8577 | 82.14% | 58.97% | 0.6866 |
| **Isolation Forest (Unsupervised)** | N/A | N/A | 82.14% | 62.16% | 0.7077 |

> 💡 **Logistic Regression** achieved the highest **Fraud Recall (92.86%)**, successfully flagging nearly all fraudulent transactions while preserving real-time inference speed.

---

## 🚀 How to Run the Project

### 1. Initialize Database & Generate Synthetic Data
```bash
python src/db_setup.py
```

### 2. Run Feature Engineering & Generate EDA Reports
```bash
python src/eda_analytics.py
```

### 3. Train Machine Learning Models & Save Pipeline
```bash
python src/train_models.py
```

### 4. Launch the Interactive Web App (Streamlit Dashboard)
```bash
streamlit run app.py
```

The web dashboard will open in your browser at `http://localhost:8501`.
