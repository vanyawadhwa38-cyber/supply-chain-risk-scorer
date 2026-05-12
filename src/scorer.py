import pandas as pd
import numpy as np
import joblib
import sys
sys.path.insert(0, 'src')
from feature_engineering import add_features
from preprocessing import preprocess

def score_suppliers(df_raw):
    model = joblib.load("models/risk_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_names.pkl")

    df = add_features(df_raw)
    df = preprocess(df)

    X = df[features]
    X_scaled = scaler.transform(X)

    disruption_prob = model.predict_proba(X_scaled)[:, 1]

    df_raw = df_raw.copy()
    df_raw["risk_score"] = (disruption_prob * 100).round(1)
    df_raw["risk_category"] = pd.cut(
        df_raw["risk_score"],
        bins=[0, 33, 66, 100],
        labels=["Low", "Medium", "High"]
    )

    return df_raw[["supplier_id", "supplier_name", "country", "region", "risk_score", "risk_category"]]

if __name__ == "__main__":
    df = pd.read_csv("data/raw/supply_chain_100k.csv")
    scored = score_suppliers(df)
    scored.to_csv("data/processed/scored_suppliers.csv", index=False)
    print("Scoring done!")
    print(scored["risk_category"].value_counts())
    print(scored.head(10))