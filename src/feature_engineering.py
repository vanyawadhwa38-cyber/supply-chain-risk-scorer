import pandas as pd

def add_features(df):
    df = df.copy()

    df["operational_risk"] = (1 - df["on_time_delivery_rate"]) * df["defect_rate"] * 100
    df["geo_logistics_risk"] = df["geopolitical_risk_index"] * df["port_congestion_score"]
    df["resilience_score"] = df["num_logistics_partners"] * df["financial_stability_score"]
    df["disruption_history_index"] = df["past_disruptions_count"] / (df["last_disruption_days_ago"] + 1) * 1000
    df["compliance_composite"] = df["compliance_score"] * df["audit_passed"]

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/raw/supply_chain_100k.csv")
    df = add_features(df)
    print("Feature engineering done!")
    print(f"Shape: {df.shape}")
    print(df[["operational_risk", "geo_logistics_risk", "resilience_score"]].head())