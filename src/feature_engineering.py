import os
import sqlite3
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "banking_fraud.db")

def haversine_np(lon1, lat1, lon2, lat2):
    """Calculates the great circle distance between two points in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def extract_features(db_path=DB_PATH):
    """Loads relational data from SQLite and computes engineered ML features."""
    conn = sqlite3.connect(db_path)

    # Load tables
    tx_df = pd.read_sql("SELECT * FROM Transaction_record", conn)
    loc_df = pd.read_sql("SELECT * FROM Transaction_location", conn)
    acc_df = pd.read_sql("SELECT * FROM accounts_details", conn)
    susp_df = pd.read_sql("SELECT * FROM Suspicious_transaction", conn)
    conn.close()

    # Clean & Cast numeric types explicitly
    tx_df['amount'] = pd.to_numeric(tx_df['amount'], errors='coerce').fillna(0.0)
    acc_df['balance'] = pd.to_numeric(acc_df['balance'], errors='coerce').fillna(0.0)
    loc_df['latitude'] = pd.to_numeric(loc_df['latitude'], errors='coerce').fillna(0.0)
    loc_df['longitude'] = pd.to_numeric(loc_df['longitude'], errors='coerce').fillna(0.0)

    # Convert timestamps
    tx_df['transaction_date'] = pd.to_datetime(tx_df['transaction_date'])

    # Merge location and account data
    df = tx_df.merge(loc_df[['transaction_id', 'latitude', 'longitude', 'city']], on='transaction_id', how='left')
    df = df.merge(acc_df[['account_id', 'balance', 'account_type']], on='account_id', how='left')

    # Merge Target Variable `is_fraud`
    df = df.merge(susp_df[['transaction_id', 'is_fraud', 'risk_score']], on='transaction_id', how='left')
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
    df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce').fillna(20).astype(int)

    # Ensure account_type is string
    df['account_type'] = df['account_type'].fillna('Unknown').astype(str)

    # Sort by customer and transaction date
    df = df.sort_values(by=['account_id', 'transaction_date']).reset_index(drop=True)

    # 1. Temporal Features
    df['hour_of_day'] = df['transaction_date'].dt.hour.astype(int)
    df['day_of_week'] = df['transaction_date'].dt.dayofweek.astype(int)
    df['is_night_transaction'] = df['hour_of_day'].apply(lambda x: 1 if 1 <= x <= 4 else 0).astype(int)

    # 2. Transaction Velocity Features (using rolling windows per account)
    tx_counts_10m = []
    tx_counts_1h = []
    tx_counts_24h = []
    tx_sums_1h = []
    travel_speeds = []

    for account_id, group in df.groupby('account_id'):
        times = group['transaction_date'].values
        amounts = group['amount'].values
        lats = group['latitude'].values
        lons = group['longitude'].values

        n = len(group)
        for i in range(n):
            curr_t = times[i]
            
            # Count transactions in windows preceding current transaction
            t_10m = curr_t - np.timedelta64(10, 'm')
            t_1h = curr_t - np.timedelta64(1, 'h')
            t_24h = curr_t - np.timedelta64(24, 'h')

            idx_10m = int(np.sum((times[:i] >= t_10m) & (times[:i] < curr_t)))
            idx_1h = int(np.sum((times[:i] >= t_1h) & (times[:i] < curr_t)))
            idx_24h = int(np.sum((times[:i] >= t_24h) & (times[:i] < curr_t)))
            
            sum_1h = float(np.sum(amounts[:i][(times[:i] >= t_1h) & (times[:i] < curr_t)])) if i > 0 else 0.0

            tx_counts_10m.append(idx_10m)
            tx_counts_1h.append(idx_1h)
            tx_counts_24h.append(idx_24h)
            tx_sums_1h.append(sum_1h)

            # Spatial Velocity / Speed calculation
            if i > 0:
                prev_lat, prev_lon, prev_t = lats[i-1], lons[i-1], times[i-1]
                dist_km = haversine_np(prev_lon, prev_lat, lons[i], lats[i])
                time_diff_hours = (curr_t - prev_t) / np.timedelta64(1, 'h')
                speed = float(dist_km / time_diff_hours) if time_diff_hours > 0 else 0.0
            else:
                speed = 0.0

            travel_speeds.append(speed)

    df['tx_count_10m'] = np.array(tx_counts_10m, dtype=int)
    df['tx_count_1h'] = np.array(tx_counts_1h, dtype=int)
    df['tx_count_24h'] = np.array(tx_counts_24h, dtype=int)
    df['tx_sum_1h'] = np.array(tx_sums_1h, dtype=float)
    df['travel_speed_kmh'] = np.clip(np.array(travel_speeds, dtype=float), 0, 3000)

    # 3. Amount Ratios
    cust_avg_amount = df.groupby('account_id')['amount'].transform('mean')
    df['amount_to_avg_ratio'] = (df['amount'] / (cust_avg_amount + 1e-5)).astype(float)
    df['amount_to_balance_ratio'] = (df['amount'] / (df['balance'] + 1e-5)).astype(float)

    # 4. Device & Geo Risk Indicators
    df['impossible_speed_flag'] = (df['travel_speed_kmh'] > 800).astype(int)
    df['transaction_type_id'] = pd.to_numeric(df['transaction_type_id'], errors='coerce').fillna(1).astype(int)

    # Select final ML feature matrix
    feature_cols = [
        'amount', 'balance', 'account_type', 'transaction_type_id',
        'hour_of_day', 'day_of_week', 'is_night_transaction',
        'tx_count_10m', 'tx_count_1h', 'tx_count_24h', 'tx_sum_1h',
        'travel_speed_kmh', 'amount_to_avg_ratio', 'amount_to_balance_ratio',
        'impossible_speed_flag'
    ]

    print("Feature Engineering completed successfully.")
    print(f"Extracted {len(feature_cols)} features across {len(df)} transactions.")
    return df, feature_cols

if __name__ == "__main__":
    df, features = extract_features()
    print("Data types:")
    print(df[features].dtypes)
