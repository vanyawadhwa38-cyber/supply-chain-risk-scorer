import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess(df):
    df = df.copy()

    drop_cols = ["supplier_id", "supplier_name", "country"]
    df = df.drop(columns=drop_cols)

    le = LabelEncoder()
    df["region"] = le.fit_transform(df["region"])

    df = df.drop(columns=["raw_risk_score", "risk_label"])

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/supply_chain_100k.csv")
    processed = preprocess(df)
    os.makedirs("data/processed", exist_ok=True)
    processed.to_csv("data/processed/supply_chain_processed.csv", index=False)
    print("Preprocessing done!")
    print(f"Shape: {processed.shape}")
    print(processed.head())