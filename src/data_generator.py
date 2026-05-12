import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

N = 100_000

def generate_data(n=N):
    data = {
        "supplier_id": [f"SUP-{i:06d}" for i in range(n)],
        "supplier_name": [fake.company() for _ in range(n)],
        "country": [fake.country() for _ in range(n)],
        "region": np.random.choice(["Asia", "Europe", "Americas", "Middle East", "Africa"], n),
        "lead_time_days": np.random.normal(30, 10, n).clip(5, 90).astype(int),
        "on_time_delivery_rate": np.random.beta(8, 2, n),
        "defect_rate": np.random.beta(2, 20, n),
        "order_fulfillment_rate": np.random.beta(9, 1, n),
        "financial_stability_score": np.random.uniform(1, 10, n),
        "years_in_business": np.random.randint(1, 50, n),
        "geopolitical_risk_index": np.random.uniform(0, 10, n),
        "natural_disaster_exposure": np.random.uniform(0, 10, n),
        "trade_restriction_flag": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "shipping_delay_days_avg": np.random.exponential(3, n).clip(0, 30),
        "port_congestion_score": np.random.uniform(0, 10, n),
        "num_logistics_partners": np.random.randint(1, 20, n),
        "past_disruptions_count": np.random.poisson(2, n),
        "last_disruption_days_ago": np.random.randint(0, 1000, n),
        "compliance_score": np.random.uniform(1, 10, n),
        "audit_passed": np.random.choice([0, 1], n, p=[0.2, 0.8]),
    }

    df = pd.DataFrame(data)

    risk_score = (
        (1 - df["on_time_delivery_rate"]) * 25 +
        df["defect_rate"] * 20 +
        df["geopolitical_risk_index"] / 10 * 20 +
        df["past_disruptions_count"] / 10 * 15 +
        df["shipping_delay_days_avg"] / 30 * 10 +
        df["trade_restriction_flag"] * 10
    )

    df["raw_risk_score"] = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 100

    df["risk_label"] = pd.cut(
        df["raw_risk_score"],
        bins=[0, 33, 66, 100],
        labels=["Low", "Medium", "High"]
    )

    df["disruption_occurred"] = (df["raw_risk_score"] > 60).astype(int)

    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/raw/supply_chain_100k.csv", index=False)
    print(f"Generated {len(df)} records")
    print(df["risk_label"].value_counts())