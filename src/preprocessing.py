"""
preprocessing.py

Data cleaning and preprocessing functions for satellite imagery and tabular features.
"""

import pandas as pd
import numpy as np

def mask_s2_clouds(image):
    """Cloud/shadow masking using QA60 and SCL for Sentinel-2."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask1 = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
              .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    # SCL logic can be added here
    return image.updateMask(mask1).divide(10000)

def mask_landsat_clouds(image):
    """Landsat QA_PIXEL masking for clouds and shadows."""
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit_mask = 1 << 4
    clouds_bit_mask = 1 << 3
    mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
             .And(qa.bitwiseAnd(clouds_bit_mask).eq(0))
    return image.updateMask(mask).multiply(0.0000275).add(-0.2)

def compute_dekadal_composite(collection, year: int, month: int, dekad: int, method: str = 'median'):
    """Creates a 10-day composite from an ImageCollection."""
    # Placeholder for GEE implementation
    return collection.median()

def compute_monthly_composite(collection, year: int, month: int):
    """Creates a monthly composite from an ImageCollection."""
    return collection.median()

def harmonize_to_grid(raster_data, target_resolution_deg: float = 0.01):
    """Reprojects and resamples raster data to a standard grid."""
    return raster_data

def compute_rainfall_windows(daily_rain: pd.Series, windows: list = [7, 14, 30, 60]) -> pd.DataFrame:
    """Computes rolling accumulations for rainfall over different windows."""
    df = pd.DataFrame()
    for w in windows:
        df[f'rain_{w}d'] = daily_rain.rolling(window=w, min_periods=1).sum()
    return df

def compute_rainfall_anomaly(current_rain: pd.Series, climatological_mean: pd.Series, climatological_std: pd.Series) -> pd.Series:
    """Computes rainfall anomaly as a z-score."""
    return (current_rain - climatological_mean) / climatological_std.replace(0, np.nan)

def track_missingness(data: pd.DataFrame, required_cols: list) -> float:
    """Returns the percentage of valid observations across required columns."""
    valid_mask = data[required_cols].notna().all(axis=1)
    return valid_mask.mean() * 100

def apply_quality_filter(df: pd.DataFrame, min_obs_pct: float = 0.5) -> pd.DataFrame:
    """Drops cells with too few valid observations."""
    obs_count = df.groupby('cell_id').size()
    max_obs = obs_count.max()
    valid_cells = obs_count[obs_count >= (max_obs * min_obs_pct)].index
    return df[df['cell_id'].isin(valid_cells)]
