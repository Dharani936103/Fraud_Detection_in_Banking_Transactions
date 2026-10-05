import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from feature_engineering import extract_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "fraud_detection_pipeline.joblib")

def train_and_evaluate_models():
    """Trains baseline, tree ensemble, XGBoost, and unsupervised Isolation Forest models."""
    df, feature_cols = extract_features()

    X = df[feature_cols]
    y = df['is_fraud']

    # Identify numerical and categorical columns
    num_cols = [col for col in feature_cols if col != 'account_type']
    cat_cols = ['account_type']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Train/Test Split (Stratified 80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nData Split: Train size = {len(X_train)}, Test size = {len(X_test)}")
    print(f"Train Class Distribution: Normal = {sum(y_train==0)}, Fraud = {sum(y_train==1)}")

    # Preprocess Features
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Apply SMOTE to handle Class Imbalance on Training Set
    print("\nApplying SMOTE for Class Imbalance Balancing...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    print(f"Balanced Train Size: Normal = {sum(y_train_res==0)}, Fraud = {sum(y_train_res==1)}")

    models = {
        'Logistic Regression (Baseline)': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost Classifier': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION METRICS")
    print("="*60)

    for name, model in models.items():
        # Train model
        model.fit(X_train_res, y_train_res)

        # Predict
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        fraud_f1 = report['1']['f1-score']

        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': fraud_f1,
            'cm': cm
        }

        print(f"\n--- {name} ---")
        print(f"ROC-AUC Score: {roc_auc:.4f} | PR-AUC Score: {pr_auc:.4f}")
        print(f"Fraud Recall: {report['1']['recall']:.4f} | Fraud Precision: {report['1']['precision']:.4f} | F1-Score: {fraud_f1:.4f}")
        print("Confusion Matrix:")
        print(f"  [TN: {cm[0][0]}, FP: {cm[0][1]}]")
        print(f"  [FN: {cm[1][0]}, TP: {cm[1][1]}]")

        if fraud_f1 > best_f1:
            best_f1 = fraud_f1
            best_model_name = name

    # Unsupervised Anomaly Detection: Isolation Forest
    print("\n--- Isolation Forest (Unsupervised Anomaly Detection) ---")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train_proc)
    iso_preds = iso_forest.predict(X_test_proc)
    # IsolationForest outputs 1 for inliers (normal) and -1 for outliers (fraud)
    iso_preds_binary = np.where(iso_preds == -1, 1, 0)
    iso_cm = confusion_matrix(y_test, iso_preds_binary)
    iso_report = classification_report(y_test, iso_preds_binary, output_dict=True)
    print(f"Isolation Forest Fraud Recall: {iso_report['1']['recall']:.4f} | Precision: {iso_report['1']['precision']:.4f}")
    print("Confusion Matrix:")
    print(f"  [TN: {iso_cm[0][0]}, FP: {iso_cm[0][1]}]")
    print(f"  [FN: {iso_cm[1][0]}, TP: {iso_cm[1][1]}]")

    print("\n" + "="*60)
    print(f"BEST SUPERVISED MODEL: {best_model_name} (F1-Score = {best_f1:.4f})")
    print("="*60)

    # Save Pipeline with Preprocessor & Best Model
    best_estimator = results[best_model_name]['model']
    full_pipeline = {
        'preprocessor': preprocessor,
        'model': best_estimator,
        'feature_cols': feature_cols,
        'model_name': best_model_name,
        'metrics': results[best_model_name]
    }
    
    joblib.dump(full_pipeline, MODEL_PATH)
    print(f"\nModel pipeline successfully saved to: {MODEL_PATH}")

    return results

if __name__ == "__main__":
    train_and_evaluate_models()
