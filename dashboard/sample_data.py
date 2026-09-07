import numpy as np
import random
from datetime import datetime, timedelta

def generate_spatial_data():
    np.random.seed(42)
    zones = ['Borena', 'East Hararghe', 'West Hararghe', 'Guji', 'Bale', 'Arsi', 'Jimma', 'Illubabor']
    
    # Oromia bounding box approx: 38.5 to 40.5 Lon, 7.5 to 9.5 Lat
    lats = np.random.uniform(7.5, 9.5, 200)
    lons = np.random.uniform(38.5, 40.5, 200)
    
    data = []
    for i in range(200):
        # Create realistic spatial pattern (lower lats tend to be drier in this sample)
        lat = lats[i]
        lon = lons[i]
        
        # simulated dryness factor based on lat/lon
        dry_factor = ((9.5 - lat) / 2.0) * 0.7 + ((lon - 38.5) / 2.0) * 0.3
        
        vci = max(10, min(100, 80 - (dry_factor * 60) + np.random.normal(0, 10)))
        ndvi = max(0.1, min(0.9, 0.7 - (dry_factor * 0.4) + np.random.normal(0, 0.05)))
        
        if vci > 50:
            risk_class = "Normal"
            risk_score = np.random.uniform(0, 0.2)
        elif vci > 35:
            risk_class = "Watch"
            risk_score = np.random.uniform(0.2, 0.5)
        elif vci > 20:
            risk_class = "Warning"
            risk_score = np.random.uniform(0.5, 0.8)
        else:
            risk_class = "Severe"
            risk_score = np.random.uniform(0.8, 1.0)
            
        data.append({
            "id": i,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "zone": random.choice(zones),
            "vci": round(vci, 2),
            "ndvi": round(ndvi, 3),
            "risk_score": round(risk_score, 2),
            "risk_class": risk_class
        })
    return data

def generate_timeseries_data(zone):
    np.random.seed(hash(zone) % (2**32))
    
    start_date = datetime(2018, 1, 1)
    dates = [(start_date + timedelta(days=30*i)).strftime('%Y-%m') for i in range(84)] # 7 years
    
    # Base pattern
    base_ndvi = 0.5
    base_vci = 50
    base_rain = 80
    base_sm = 0.3
    
    ndvi = []
    vci = []
    rainfall = []
    soil_moisture = []
    risk_score = []
    
    for i, date in enumerate(dates):
        # Seasonality (peaks in summer months around August)
        month = int(date.split('-')[1])
        season = np.sin((month - 5) * np.pi / 6)
        
        # Drought event in 2021-2022
        year = int(date.split('-')[0])
        drought_factor = 1.0
        if year in [2021, 2022]:
            drought_factor = 0.4
            
        n = base_ndvi + (season * 0.2) * drought_factor + np.random.normal(0, 0.05)
        v = base_vci + (season * 20) * drought_factor + np.random.normal(0, 5)
        if year in [2021, 2022]:
            v -= 30
            
        r = max(0, base_rain + (season * 50)) * drought_factor + np.random.normal(0, 10)
        sm = base_sm + (season * 0.1) * drought_factor + np.random.normal(0, 0.02)
        
        n = max(0, min(1, n))
        v = max(0, min(100, v))
        sm = max(0, min(1, sm))
        
        rs = 1.0 - (v / 100.0)
        
        ndvi.append(round(n, 3))
        vci.append(round(v, 2))
        rainfall.append(round(r, 2))
        soil_moisture.append(round(sm, 3))
        risk_score.append(round(rs, 2))
        
    return {
        "dates": dates,
        "ndvi": ndvi,
        "vci": vci,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "risk_score": risk_score
    }

def generate_model_results():
    return {
        "comparison": [
            {"model": "Rule Baseline", "F1": 0.55, "FAR": 0.42, "POD": 0.60, "RMSE": 18.5, "Lead10": 0.55, "Lead20": 0.48, "Lead30": 0.40},
            {"model": "Random Forest", "F1": 0.82, "FAR": 0.15, "POD": 0.85, "RMSE": 8.2, "Lead10": 0.80, "Lead20": 0.72, "Lead30": 0.61},
            {"model": "XGBoost", "F1": 0.86, "FAR": 0.12, "POD": 0.89, "RMSE": 7.5, "Lead10": 0.84, "Lead20": 0.78, "Lead30": 0.68},
            {"model": "LSTM", "F1": 0.84, "FAR": 0.14, "POD": 0.86, "RMSE": 7.8, "Lead10": 0.81, "Lead20": 0.79, "Lead30": 0.74}
        ],
        "feature_importance": {
            "features": ["NDVI", "VCI", "Rainfall (1mo)", "Soil Moisture", "LST", "NDMI", "EVI", "Rainfall (3mo)"],
            "importance": [0.28, 0.22, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04]
        },
        "lead_time_skill": {
            "leads": [10, 20, 30, 40, 50, 60],
            "xgboost": [0.86, 0.78, 0.68, 0.55, 0.42, 0.35],
            "lstm": [0.84, 0.79, 0.74, 0.62, 0.55, 0.48]
        }
    }

def generate_alerts():
    zones = ['Borena', 'East Hararghe', 'West Hararghe', 'Guji', 'Bale', 'Arsi', 'Jimma', 'Illubabor']
    alerts = []
    
    for i in range(20):
        level = random.choices(['Severe', 'Warning', 'Watch', 'Normal'], weights=[10, 30, 40, 20])[0]
        
        confidence = random.randint(75, 95)
        lead_time = random.choice([10, 20, 30])
        
        alerts.append({
            "id": f"ALT-{random.randint(1000, 9999)}",
            "zone": random.choice(zones),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "level": level,
            "confidence": confidence,
            "lead_time_days": lead_time,
            "message": f"[{level.upper()}] Drought risk detected in {random.choice(zones)}. Confidence: {confidence}%. Action recommended within {lead_time} days."
        })
        
    # Sort so severe are first
    alerts.sort(key=lambda x: ['Severe', 'Warning', 'Watch', 'Normal'].index(x['level']))
    return alerts
