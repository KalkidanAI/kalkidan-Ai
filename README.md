# 🛰️ From Space to Action: AI + Satellite Data for Agricultural Drought Early Warning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KalkidanAI/kalkidan-Ai/blob/main/notebooks/train_all_models_colab.ipynb)

**Author:** Kalkidan Wondimagegnehu Alemneh  
**Institution:** EGATE (School of Gifted and Talented)  
**Study Area:** Oromia Region, Ethiopia (Arsi-Bale Pilot Zone)  

---

## 📌 Project Overview

This project builds an operational, multimodal machine learning and deep learning pipeline to forecast agricultural drought at **10, 20, and 30-day lead horizons** across the Oromia region of Ethiopia.

The pipeline integrates multi-source satellite earth observation and climate data:
- **Optical Vegetation Indices**: Sentinel-2 ($NDVI, EVI, NDMI$) & Landsat 8/9
- **Vegetation Health**: Vegetation Condition Index ($VCI$)
- **Precipitation**: CHIRPS daily rainfall (30-day and 60-day accumulations, anomalies)
- **Soil Moisture**: NASA/USDA SMAP 10km root-zone soil moisture
- **Thermal & Topography**: ERA5-Land land surface temperature & SRTM elevation

---

## ⚡ 1-Click Google Colab Training

Click the badge above or use this direct link to train the models in Google Colab:
👉 **[Open Master Colab Notebook](https://colab.research.google.com/github/KalkidanAI/kalkidan-Ai/blob/main/notebooks/train_all_models_colab.ipynb)**

### What the Master Notebook Runs:
1. **Model 0 (Baseline)**: Operational VCI threshold model ($VCI \le 35$).
2. **Model 1 (Random Forest)**: Balanced ensemble with feature importance ranking.
3. **Model 2 (XGBoost + Ablation + SHAP)**: Gradient-boosted decision trees with modality ablation and Shapley explainability.
4. **Model 3 (PyTorch LSTM)**: Temporal recurrent neural network forecasting sequential drought evolution on GPU.
5. **Multi-Horizon Evaluation**: Evaluating skill across 10-day, 20-day, and 30-day forecast horizons.
6. **Zero-Leakage Audit**: Verification that train ($\le 2022$), validation ($2022–2023.5$), and test ($> 2023.5$) splits are strictly disjoint.
7. **Automated Export**: Downloads `trained_models.zip` directly to your computer for web dashboard deployment.

---

## 📁 Repository Structure

```
├── config/
│   └── config.yaml               # Study area boundaries, dates, model hyperparameters
├── dashboard/                    # Interactive web dashboard (Flask)
│   ├── app.py                   # API routes and server
│   ├── sample_data.py           # Fallback mock generator
│   ├── static/                  # CSS & JavaScript assets
│   └── templates/               # HTML5 templates (maps, analytics, models, alerts)
├── data/
│   └── targets/
│       ├── train.csv            # 146,000 observations (<= 2022-01-01)
│       ├── val.csv              # 55,000 observations (2022-01-01 to 2023-07-01)
│       └── test.csv             # 91,000 observations (> 2023-07-01, held out)
├── notebooks/
│   ├── train_all_models_colab.ipynb  # 🚀 Master Colab Training Pipeline
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_target_construction.ipynb
│   ├── 05_eda_historical.ipynb
│   ├── 06_baseline_rule_model.ipynb
│   ├── 07_random_forest.ipynb
│   ├── 08_xgboost.ipynb
│   ├── 09_lstm_forecasting.ipynb
│   ├── 10_model_comparison.ipynb
│   ├── 11_dashboard_maps.ipynb
│   └── 12_alert_sms_prototype.ipynb
├── scripts/
│   ├── build_colab_notebooks.py # Synchronizes and rebuilds all notebooks
│   ├── nb_builder.py            # Programmatic Jupyter notebook constructor
│   └── prepare_dataset.py       # Pilot data generator & temporal splitter
├── src/
│   ├── alert_generator.py       # SMS and dashboard alert formatting
│   ├── data_loader.py           # Earth Engine data acquisition routines
│   ├── drought_index.py         # VCI, TCI, and CADI computation
│   ├── evaluation.py            # FAR, POD, Macro F1, confusion matrices
│   ├── features.py              # NDVI, EVI, NDMI, rolling means, lags, trends
│   ├── models.py                # RuleBaseline, DroughtRF, DroughtXGB, DroughtLSTM
│   ├── preprocessing.py         # Cloud masking and dekadal compositing
│   └── visualization.py         # Map plotting and charting routines
├── README_COLAB.md              # Detailed Colab step-by-step documentation
└── requirements.txt             # Python dependencies
```

---

## 💻 Local Setup & Running Dashboard

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run web dashboard
python dashboard/app.py
```
Open `http://localhost:5000` to view the interactive spatial risk map and model benchmarks.
