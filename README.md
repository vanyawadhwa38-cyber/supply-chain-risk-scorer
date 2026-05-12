# 🔗 Supply Chain Disruption Risk Scorer

An end-to-end Machine Learning project that analyzes **100,000 global supplier records** to predict and score supply chain disruption risk.

## 📊 Project Overview

This project builds a complete supplier risk scoring system that:
- Generates and processes 100,000 synthetic supplier records
- Engineers 23+ features across multiple risk dimensions
- Trains an XGBoost ML model with **99% ROC-AUC score**
- Scores every supplier with a risk probability (0-100)
- Displays results in an interactive business dashboard

## 🚀 Key Results

| Metric | Value |
|--------|-------|
| Records Analyzed | 100,000 |
| Model Accuracy | 95% |
| ROC-AUC Score | 0.9937 |
| Features Engineered | 23 |
| Risk Categories | High / Medium / Low |

## 🛠️ Tools & Technologies

- **Python** — Core programming language
- **XGBoost** — ML model for disruption prediction
- **Scikit-learn** — Preprocessing and evaluation
- **SMOTE** — Handling class imbalance
- **Plotly Dash** — Interactive business dashboard
- **Pandas & NumPy** — Data manipulation

## 📁 Project Structure

supply_chain_risk/
├── data/
│   ├── raw/                   # Generated 100k records
│   └── processed/             # Cleaned and scored data
├── src/
│   ├── data_generator.py      # Generates 100,000 supplier records
│   ├── preprocessing.py       # Cleans and encodes data
│   ├── feature_engineering.py # Creates composite risk features
│   ├── model.py               # Trains XGBoost model
│   └── scorer.py              # Scores all suppliers
├── dashboards/
│   └── app.py                 # Plotly Dash dashboard
└── requirements.txt

## ⚙️ How to Run

pip install -r requirements.txt
python src/data_generator.py
python src/model.py
python src/scorer.py
python dashboards/app.py

Then open http://127.0.0.1:8050 in your browser.

## 📈 Risk Dimensions Analyzed

- **Operational** — delivery rate, defect rate, lead time
- **Financial** — stability score, years in business
- **Geopolitical** — risk index, trade restrictions
- **Logistics** — shipping delays, port congestion
- **Historical** — past disruptions, compliance score