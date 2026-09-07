"""
visualization.py

Mapping, plotting, and dashboard utilities.
"""

import pandas as pd

DROUGHT_COLORS = {
    0: 'green',
    1: 'yellow',
    2: 'orange',
    3: 'red'
}

def plot_oromia_drought_map(gdf, risk_col: str, title: str):
    """Folium choropleth map placeholder."""
    pass

def plot_timeseries(df: pd.DataFrame, cols: list, title: str, date_col: str = 'date'):
    """Plotly time-series plotting placeholder."""
    pass

def plot_feature_importance(importances, feature_names, top_n: int = 20):
    """Horizontal bar chart for feature importances."""
    pass

def plot_drought_history(df: pd.DataFrame, cell_id, features: list = ['ndvi', 'vci', 'rain_30d']):
    """Multi-panel matplotlib history plot."""
    pass

def plot_spatial_risk(lat, lon, risk, title: str):
    """Scatter map with risk colors."""
    pass

def plot_model_comparison(results_df: pd.DataFrame):
    """Grouped bar chart for model performance comparisons."""
    pass

def plot_lead_time_skill(leads: list, metrics_by_lead: list):
    """Line chart showing degradation of skill over lead time."""
    pass

def create_dashboard_html(data: pd.DataFrame, output_path: str):
    """Standalone HTML dashboard export."""
    html_content = f"<html><body><h1>Drought Dashboard</h1><p>Data rows: {len(data)}</p></body></html>"
    with open(output_path, 'w') as f:
        f.write(html_content)
