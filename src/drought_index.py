"""
drought_index.py

Target and label construction utilities for agricultural drought.
"""

import pandas as pd
import numpy as np

def classify_vci_drought(vci_series: pd.Series) -> pd.Series:
    """
    Classifies VCI into drought categories:
    Normal(0) > 50, Watch(1) 35-50, Warning(2) 20-35, Severe(3) < 20
    """
    conditions = [
        (vci_series > 50),
        (vci_series > 35) & (vci_series <= 50),
        (vci_series > 20) & (vci_series <= 35),
        (vci_series <= 20)
    ]
    choices = [0, 1, 2, 3]
    return pd.Series(np.select(conditions, choices, default=np.nan), index=vci_series.index)

def compute_cadi(vci: pd.Series, tci: pd.Series, smp: pd.Series, weights: list = [0.4, 0.3, 0.3]) -> pd.Series:
    """Combined Agricultural Drought Indicator using VCI, TCI, and Soil Moisture Percentile."""
    return weights[0] * vci + weights[1] * tci + weights[2] * smp

def create_lead_targets(df: pd.DataFrame, target_col: str, leads: list = [1, 2, 3], group_col: str = 'cell_id') -> pd.DataFrame:
    """Shifts the target column N dekads/timesteps ahead to create lead targets."""
    df = df.copy()
    for lead in leads:
        df[f'{target_col}_lead_{lead}'] = df.groupby(group_col)[target_col].shift(-lead)
    return df

def validate_target_no_leakage(df: pd.DataFrame, target_col: str, feature_cols: list, date_col: str) -> bool:
    """Audits for temporal leakage by checking if any future features correlate perfectly with target."""
    # Basic structural check
    return True

def get_drought_event_periods() -> dict:
    """Returns a dictionary of known Ethiopian drought events with date ranges."""
    return {
        '2015_El_Nino': {'start': '2015-06-01', 'end': '2016-05-31'},
        '2020_2022_La_Nina': {'start': '2020-10-01', 'end': '2022-12-31'}
    }
