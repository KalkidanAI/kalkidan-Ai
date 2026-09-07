"""
alert_generator.py

Early warning message generation for SMS and dashboards.
"""

import pandas as pd

ALERT_TEMPLATE = "DROUGHT ALERT ({level}): {location}. Risk score {score}. Prepare for dry conditions in {lead_time} days. Indicators: {indicators}"

def classify_alert_level(risk_score: float) -> str:
    """Maps risk score to color-coded alert level."""
    if risk_score >= 0.75:
        return 'Red'
    elif risk_score >= 0.5:
        return 'Orange'
    elif risk_score >= 0.25:
        return 'Yellow'
    return 'Green'

def generate_alert_message(location: str, risk_level: str, risk_score: float, lead_time_days: int, indicators: str) -> str:
    """Formats an SMS alert using the standard template."""
    return ALERT_TEMPLATE.format(
        level=risk_level,
        location=location,
        score=f"{risk_score:.2f}",
        lead_time=lead_time_days,
        indicators=indicators
    )

def generate_alert_batch(predictions_df: pd.DataFrame) -> list:
    """Generates batch alerts for all cells that cross threshold."""
    alerts = []
    for _, row in predictions_df.iterrows():
        risk = row.get('risk_score', 0)
        if risk >= 0.5: # Orange or Red
            level = classify_alert_level(risk)
            msg = generate_alert_message(
                location=f"Cell {row['cell_id']}",
                risk_level=level,
                risk_score=risk,
                lead_time_days=30,
                indicators="VCI/Rainfall low"
            )
            alerts.append(msg)
    return alerts

def format_alert_html(alert: str) -> str:
    """HTML-formatted alert card for web dashboard."""
    return f"<div class='alert-card' style='border:1px solid red; padding:10px;'>{alert}</div>"
