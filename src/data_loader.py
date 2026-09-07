"""
data_loader.py

Google Earth Engine helper functions for fetching satellite imagery and climate data,
along with utilities for generating synthetic sample data for offline/MVP testing.
"""

import pandas as pd
import numpy as np
import datetime

try:
    import ee
except ImportError:
    ee = None


def get_oromia_boundary():
    """Returns ee.Geometry for Oromia from FAO GAUL."""
    if ee is None: return None
    return ee.FeatureCollection("FAO/GAUL/2015/level1") \
             .filter(ee.Filter.eq('ADM1_NAME', 'Oromia')) \
             .geometry()

def get_pilot_aoi(bbox: list):
    """Creates ee.Geometry.Rectangle from bbox [minLon, minLat, maxLon, maxLat]."""
    if ee is None: return None
    return ee.Geometry.Rectangle(bbox)

def fetch_sentinel2(aoi, start: str, end: str, max_cloud: int = 30):
    """Fetches filtered Sentinel-2 collection."""
    if ee is None: return None
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
             .filterBounds(aoi) \
             .filterDate(start, end) \
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))

def fetch_landsat(aoi, start: str, end: str):
    """Fetches combined Landsat 8/9 collection."""
    if ee is None: return None
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    combined = l8.merge(l9)
    return combined.filterBounds(aoi).filterDate(start, end)

def fetch_chirps(aoi, start: str, end: str):
    """Fetches CHIRPS daily rainfall."""
    if ee is None: return None
    return ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
             .filterBounds(aoi) \
             .filterDate(start, end)

def fetch_era5(aoi, start: str, end: str):
    """Fetches ERA5-Land monthly."""
    if ee is None: return None
    return ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
             .filterBounds(aoi) \
             .filterDate(start, end)

def fetch_smap(aoi, start: str, end: str):
    """Fetches SMAP soil moisture."""
    if ee is None: return None
    return ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture") \
             .filterBounds(aoi) \
             .filterDate(start, end)

def fetch_dem(aoi):
    """Fetches SRTM elevation."""
    if ee is None: return None
    return ee.Image("USGS/SRTMGL1_003").clip(aoi)

def fetch_landcover(aoi):
    """Fetches ESA WorldCover."""
    if ee is None: return None
    return ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)

def export_to_drive(collection, description: str, folder: str, scale: int, region):
    """Batch exports GEE collection to Google Drive."""
    if ee is None: return None
    task = ee.batch.Export.image.toDrive(
        image=collection,
        description=description,
        folder=folder,
        scale=scale,
        region=region
    )
    task.start()
    return task

def generate_sample_data(n_cells: int = 5000, n_timesteps: int = 72) -> pd.DataFrame:
    """Generates realistic synthetic tabular data for MVP/demo when GEE is unavailable."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_timesteps, freq='10D')
    
    records = []
    for cell_id in range(n_cells):
        lat = np.random.uniform(7.5, 9.5)
        lon = np.random.uniform(38.5, 40.5)
        elevation = np.random.uniform(1000, 3000)
        
        # Base seasonal pattern
        seasonality = np.sin(np.linspace(0, 4 * np.pi, n_timesteps))
        
        for i, date in enumerate(dates):
            ndvi = np.clip(0.4 + 0.2 * seasonality[i] + np.random.normal(0, 0.05), 0, 1)
            evi = ndvi * 0.8
            ndmi = ndvi * 0.7 + np.random.normal(0, 0.02)
            vci = np.clip(50 + 20 * seasonality[i] + np.random.normal(0, 10), 0, 100)
            rain_30d = max(0, 50 + 50 * seasonality[i] + np.random.normal(0, 20))
            rain_anomaly = np.random.normal(0, 1)
            smap_sm = np.clip(0.2 + 0.1 * seasonality[i] + np.random.normal(0, 0.05), 0, 0.5)
            smap_anomaly = np.random.normal(0, 1)
            temp_anomaly = np.random.normal(0, 1)
            
            records.append({
                'cell_id': cell_id,
                'date': date,
                'lat': lat,
                'lon': lon,
                'ndvi': ndvi,
                'evi': evi,
                'ndmi': ndmi,
                'vci': vci,
                'rain_30d': rain_30d,
                'rain_anomaly': rain_anomaly,
                'smap_sm': smap_sm,
                'smap_anomaly': smap_anomaly,
                'temp_anomaly': temp_anomaly,
                'elevation': elevation
            })
            
    return pd.DataFrame(records)
