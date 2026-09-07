"""
features.py

Feature engineering pipeline for creating analysis-ready tabular data.
"""

import pandas as pd
import numpy as np

def compute_ndvi(nir: pd.Series, red: pd.Series) -> pd.Series:
    """(NIR - RED) / (NIR + RED)"""
    return (nir - red) / (nir + red)

def compute_evi(nir: pd.Series, red: pd.Series, blue: pd.Series) -> pd.Series:
    """Enhanced Vegetation Index."""
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

def compute_ndmi(nir: pd.Series, swir: pd.Series) -> pd.Series:
    """Normalized Difference Moisture Index."""
    return (nir - swir) / (nir + swir)

def compute_vci(ndvi: pd.Series, ndvi_min: pd.Series, ndvi_max: pd.Series) -> pd.Series:
    """Vegetation Condition Index."""
    return 100 * (ndvi - ndvi_min) / (ndvi_max - ndvi_min).replace(0, np.nan)

def compute_tci(lst: pd.Series, lst_min: pd.Series, lst_max: pd.Series) -> pd.Series:
    """Temperature Condition Index."""
    return 100 * (lst_max - lst) / (lst_max - lst_min).replace(0, np.nan)

def compute_vhi(vci: pd.Series, tci: pd.Series, alpha: float = 0.5) -> pd.Series:
    """Vegetation Health Index."""
    return alpha * vci + (1 - alpha) * tci

def add_temporal_features(df: pd.DataFrame, value_col: str, lags: list = [1, 2, 4], windows: list = [3, 6]) -> pd.DataFrame:
    """Adds lag and rolling window statistics for a feature."""
    df = df.copy()
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df.groupby('cell_id')[value_col].shift(lag)
    for window in windows:
        df[f'{value_col}_rollmean_{window}'] = df.groupby('cell_id')[value_col].rolling(window).mean().reset_index(0, drop=True)
    return df

def add_seasonal_encoding(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Adds cyclical encoding for month of year."""
    df = df.copy()
    month = df[date_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    return df

def add_trend_feature(df: pd.DataFrame, value_col: str, window: int = 6) -> pd.DataFrame:
    """Adds linear trend slope over a rolling window."""
    df = df.copy()
    def compute_slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        poly = np.polyfit(x, y, 1)
        return poly[0]
    
    df[f'{value_col}_trend_{window}'] = df.groupby('cell_id')[value_col].rolling(window).apply(compute_slope).reset_index(0, drop=True)
    return df

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator that builds analysis-ready DataFrame."""
    df = df.copy()
    df = add_seasonal_encoding(df, 'date')
    if 'ndvi' in df.columns:
        df = add_temporal_features(df, 'ndvi', lags=[1,2], windows=[3])
    if 'vci' in df.columns:
        df = add_temporal_features(df, 'vci', lags=[1,2], windows=[3])
        df = add_trend_feature(df, 'vci', window=4)
    return df
