import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys
sys.path.insert(0, 'src')
from feature_engineering import add_features
from preprocessing import preprocess

def train_model():
    df = pd.read_csv("data/raw/supply_chain_100k.csv")
    df = add_features(df)
    df = preprocess(df)

    X = df.drop(columns=["disruption_occurred"])
    y = df["disruption_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_scaled, y_train_res)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")
    print("Model saved!")

if __name__ == "__main__":
    train_model()