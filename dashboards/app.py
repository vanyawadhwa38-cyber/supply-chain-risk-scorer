import sys
sys.path.insert(0, 'src')
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.express as px

df = pd.read_csv("data/processed/scored_suppliers.csv")

app = dash.Dash(__name__)

total = len(df)
high = (df["risk_category"] == "High").sum()
medium = (df["risk_category"] == "Medium").sum()
low = (df["risk_category"] == "Low").sum()

app.layout = html.Div([

    html.H1("Supply Chain Disruption Risk Dashboard",
            style={"textAlign": "center", "color": "#2c3e50", "padding": "20px"}),

    html.Div([
        html.Div([html.H2(str(total)), html.P("Total Suppliers")],
                 style={"background": "#3498db", "color": "white", "padding": "20px",
                        "borderRadius": "10px", "textAlign": "center", "width": "20%"}),
        html.Div([html.H2(str(high)), html.P("High Risk")],
                 style={"background": "#e74c3c", "color": "white", "padding": "20px",
                        "borderRadius": "10px", "textAlign": "center", "width": "20%"}),
        html.Div([html.H2(str(medium)), html.P("Medium Risk")],
                 style={"background": "#f39c12", "color": "white", "padding": "20px",
                        "borderRadius": "10px", "textAlign": "center", "width": "20%"}),
        html.Div([html.H2(str(low)), html.P("Low Risk")],
                 style={"background": "#2ecc71", "color": "white", "padding": "20px",
                        "borderRadius": "10px", "textAlign": "center", "width": "20%"}),
    ], style={"display": "flex", "justifyContent": "space-around", "padding": "20px"}),

    html.Div([
        dcc.Graph(figure=px.pie(df, names="risk_category", title="Risk Distribution",
                                color="risk_category",
                                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"})),
        dcc.Graph(figure=px.histogram(df, x="risk_score", nbins=50,
                                      title="Risk Score Distribution",
                                      color_discrete_sequence=["#3498db"])),
        dcc.Graph(figure=px.box(df, x="region", y="risk_score", color="risk_category",
                                title="Risk by Region",
                                color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"})),
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.H3("Top High Risk Suppliers", style={"padding": "20px", "color": "#2c3e50"}),
    dash_table.DataTable(
        data=df[df["risk_category"] == "High"].sort_values("risk_score", ascending=False).head(50).to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["supplier_id", "supplier_name", "country", "region", "risk_score", "risk_category"]],
        page_size=15,
        style_data_conditional=[
            {"if": {"filter_query": '{risk_category} = "High"'},
             "backgroundColor": "#ffe0e0"}
        ],
        style_header={"backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"},
    )
], style={"fontFamily": "Arial", "backgroundColor": "#f8f9fa"})

if __name__ == "__main__":
    app.run(debug=True)