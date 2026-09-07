"""
prepare_dataset.py

End-to-end dataset generator and feature pipeline for From Space to Action
(Agricultural Drought Early Warning in Oromia, Ethiopia).

Generates synthetic, physically realistic satellite, climate, and soil moisture
time series for the Arsi-Bale pilot zone, creates temporal lags and lead targets,
and saves the train/val/test splits to the data/ directory.

Works in both standard Python (zero dependencies) and high-performance NumPy/Pandas.
"""

import os
import sys
import math
import random
from datetime import datetime, timedelta

def get_project_root():
    # If running inside scripts/, go up one level
    curr = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(curr, ".."))
    return root

def generate_pilot_dataset(n_cells=1500, start_year=2018, end_year=2025, seed=42):
    """
    Generates multi-temporal dekadal observations (every 10 days) across pilot grid cells.
    Models the real bi-modal Ethiopian rainfall patterns:
      - Belg season (Mar - May)
      - Kiremt season (Jun - Sep)
      - Bega dry season (Oct - Feb)
      - Historic 2020-2022 La Niña drought event
    """
    random.seed(seed)
    
    # Generate dekadal date list (approx 36 dekads per year)
    dates = []
    curr_date = datetime(start_year, 1, 5)
    end_date = datetime(end_year, 12, 31)
    while curr_date <= end_date:
        dates.append(curr_date)
        curr_date += timedelta(days=10)
        
    n_timesteps = len(dates)
    print(f"Generating dataset for {n_cells} grid cells across {n_timesteps} dekadal timesteps ({len(dates)*n_cells:,} total observations)...")
    
    # Pilot bounds: Arsi-Bale (lon 38.5 to 40.5, lat 7.5 to 9.5)
    cells = []
    for cid in range(n_cells):
        lat = 7.5 + random.random() * 2.0
        lon = 38.5 + random.random() * 2.0
        elev = 1200.0 + random.random() * 1600.0
        aridity_bias = ((9.5 - lat) / 2.0) * 0.6 + ((lon - 38.5) / 2.0) * 0.4
        cells.append({
            "cell_id": cid,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "elevation": round(elev, 1),
            "aridity_bias": aridity_bias
        })
        
    records = []
    
    for cell in cells:
        cid = cell["cell_id"]
        lat = cell["lat"]
        lon = cell["lon"]
        elev = cell["elevation"]
        bias = cell["aridity_bias"]
        
        # Track history for lags
        cell_records = []
        
        for t_idx, d in enumerate(dates):
            month = d.month
            year = d.year
            day = d.day
            d_str = d.strftime("%Y-%m-%d")
            
            # Cyclical seasonality
            month_sin = math.sin(2.0 * math.pi * month / 12.0)
            month_cos = math.cos(2.0 * math.pi * month / 12.0)
            
            # Rainfall seasonal profile in Ethiopia
            # Peaks in Kiremt (Jul-Aug) and minor peak in Belg (Apr)
            kiremt_peak = math.exp(-((month - 7.8) ** 2) / 2.5)
            belg_peak = 0.45 * math.exp(-((month - 4.2) ** 2) / 1.5)
            seasonality = kiremt_peak + belg_peak
            
            # 2020-2022 La Niña drought event
            drought_penalty = 0.0
            if year == 2020 and month >= 9:
                drought_penalty = 0.25
            elif year in (2021, 2022):
                drought_penalty = 0.45 + (0.15 * bias)
            elif year == 2023 and month <= 6:
                drought_penalty = 0.20
                
            noise = (random.random() - 0.5) * 0.1
            
            # Base indices
            ndvi_base = 0.30 + 0.45 * seasonality - (drought_penalty * 0.35) - (bias * 0.15) + noise
            ndvi = max(0.05, min(0.92, ndvi_base))
            evi = max(0.04, min(0.85, ndvi * 0.85 + (random.random() - 0.5) * 0.04))
            ndmi = max(-0.15, min(0.65, ndvi * 0.70 - 0.10 + (random.random() - 0.5) * 0.05))
            
            # VCI (Vegetation Condition Index: 0 to 100)
            # Low VCI indicates severe drought
            vci_base = 65.0 + (seasonality * 25.0) - (drought_penalty * 55.0) - (bias * 20.0) + (random.random() - 0.5) * 12.0
            vci = max(2.0, min(98.0, vci_base))
            
            # CHIRPS 30-day and 60-day rainfall (mm)
            rain_30d = max(0.0, (15.0 + seasonality * 140.0) * (1.0 - drought_penalty) + (random.random() - 0.5) * 25.0)
            rain_60d = max(0.0, rain_30d * 1.85 + (random.random() - 0.5) * 35.0)
            rain_anomaly = (rain_30d - 75.0) / 45.0
            
            # SMAP Soil moisture (m^3/m^3)
            sm_base = 0.15 + (seasonality * 0.25) * (1.0 - drought_penalty) - (bias * 0.05) + (random.random() - 0.5) * 0.04
            smap_sm = max(0.02, min(0.48, sm_base))
            smap_anomaly = (smap_sm - 0.22) / 0.08
            
            # Temperature anomaly (ERA5 Land)
            temp_anomaly = (drought_penalty * 1.8) + (bias * 0.8) + (random.random() - 0.5) * 1.2
            lst = 295.0 + (temp_anomaly * 2.5) + (1.0 - seasonality) * 6.0
            
            # Drought class at current time (0: Normal, 1: Watch, 2: Warning, 3: Severe)
            if vci <= 20.0:
                drought_class = 3  # Severe
            elif vci <= 35.0:
                drought_class = 2  # Warning
            elif vci <= 40.0:
                drought_class = 1  # Watch
            else:
                drought_class = 0  # Normal
                
            rec = {
                "cell_id": cid,
                "date": d_str,
                "lat": lat,
                "lon": lon,
                "elevation": elev,
                "month_sin": round(month_sin, 4),
                "month_cos": round(month_cos, 4),
                "ndvi": round(ndvi, 4),
                "evi": round(evi, 4),
                "ndmi": round(ndmi, 4),
                "vci": round(vci, 2),
                "rain_30d": round(rain_30d, 2),
                "rain_60d": round(rain_60d, 2),
                "rain_anomaly": round(rain_anomaly, 4),
                "smap_sm": round(smap_sm, 4),
                "smap_anomaly": round(smap_anomaly, 4),
                "temp_anomaly": round(temp_anomaly, 4),
                "lst": round(lst, 2),
                "current_class": drought_class
            }
            cell_records.append(rec)
            
        # Add lags & lead targets per cell
        for i, rec in enumerate(cell_records):
            # Lags
            rec["ndvi_lag_1"] = cell_records[i-1]["ndvi"] if i >= 1 else rec["ndvi"]
            rec["ndvi_lag_2"] = cell_records[i-2]["ndvi"] if i >= 2 else rec["ndvi_lag_1"]
            rec["ndvi_rollmean_3"] = round(sum(cell_records[j]["ndvi"] for j in range(max(0, i-2), i+1)) / (i - max(0, i-2) + 1), 4)
            
            rec["vci_lag_1"] = cell_records[i-1]["vci"] if i >= 1 else rec["vci"]
            rec["vci_lag_2"] = cell_records[i-2]["vci"] if i >= 2 else rec["vci_lag_1"]
            rec["vci_rollmean_3"] = round(sum(cell_records[j]["vci"] for j in range(max(0, i-2), i+1)) / (i - max(0, i-2) + 1), 2)
            rec["vci_trend_4"] = round((rec["vci"] - cell_records[max(0, i-3)]["vci"]) / max(1, min(3, i)), 2)
            
            # Lead targets (1 dekad = 10d, 2 dekads = 20d, 3 dekads = 30d)
            rec["target_lead_1"] = cell_records[i+1]["current_class"] if i + 1 < len(cell_records) else rec["current_class"]
            rec["target_lead_2"] = cell_records[i+2]["current_class"] if i + 2 < len(cell_records) else rec["target_lead_1"]
            rec["target_lead_3"] = cell_records[i+3]["current_class"] if i + 3 < len(cell_records) else rec["target_lead_2"]
            
            # Primary model target: 30-day forecast (lead 3)
            rec["target"] = rec["target_lead_3"]
            
            # Assign strict temporal split
            d_val = rec["date"]
            if d_val <= "2022-01-01":
                rec["split"] = "Train"
            elif d_val <= "2023-07-01":
                rec["split"] = "Val"
            else:
                rec["split"] = "Test"
                
            records.append(rec)
            
    return records


def save_dataset(records, project_dir):
    """Saves records to CSV and Parquet formats in standard project directory layout."""
    data_dir = os.path.join(project_dir, "data")
    targets_dir = os.path.join(data_dir, "targets")
    features_dir = os.path.join(data_dir, "features")
    os.makedirs(targets_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    fieldnames = list(records[0].keys())
    
    train_records = [r for r in records if r["split"] == "Train"]
    val_records = [r for r in records if r["split"] == "Val"]
    test_records = [r for r in records if r["split"] == "Test"]
    
    print(f"\nDataset Splits:")
    print(f"  - Train observations (<= 2022-01-01): {len(train_records):,}")
    print(f"  - Validation observations (2022-01-01 to 2023-07-01): {len(val_records):,}")
    print(f"  - Test observations (> 2023-07-01, HELD OUT): {len(test_records):,}")
    
    import csv
    
    def write_csv(data, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved: {path} ({len(data):,} rows)")
        
    write_csv(train_records, os.path.join(targets_dir, "train.csv"))
    write_csv(val_records, os.path.join(targets_dir, "val.csv"))
    write_csv(test_records, os.path.join(targets_dir, "test.csv"))
    write_csv(records, os.path.join(features_dir, "features_v1.csv"))
    
    # Also save parquet if pandas/pyarrow is installed
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(os.path.join(features_dir, "features_v1.parquet"), index=False)
        df.to_parquet(os.path.join(targets_dir, "dataset_v1.parquet"), index=False)
        print("Saved Parquet versions: features_v1.parquet, dataset_v1.parquet")
    except Exception as e:
        print("Note: Parquet export skipped (pandas/pyarrow not installed locally; CSV is ready).")
        
    print("\nDataset preparation complete.")


if __name__ == "__main__":
    proj_dir = get_project_root()
    # Default: 1000 cells for fast, highly realistic pilot dataset
    records = generate_pilot_dataset(n_cells=1000, start_year=2018, end_year=2025)
    save_dataset(records, proj_dir)
