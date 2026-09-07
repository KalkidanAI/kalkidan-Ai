# 🛰️ Google Colab Model Training Guide
### **Project:** From Space to Action — AI + Satellite Data for Agricultural Drought Early Warning
**Target Study Area:** Oromia Region, Ethiopia (Arsi-Bale Pilot Zone)

---

## 🚀 Quickstart: 1-Click Master Colab Training

We have created an **All-in-One Master Training Notebook** located at:
[`notebooks/train_all_models_colab.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/train_all_models_colab.ipynb)

This single notebook handles everything end-to-end:
1. Environment and GPU acceleration setup.
2. Multimodal data preparation (or loading existing pilot data).
3. **Model 0 (Baseline)**: Agronomic rule-based threshold model ($VCI \le 35$).
4. **Model 1 (Random Forest)**: 300 estimators with class balancing & feature importance.
5. **Model 2 (XGBoost + Ablation + SHAP)**: Gradient boosting with modality contribution testing & explainability.
6. **Model 3 (PyTorch LSTM)**: Sequence forecasting on GPU with sliding temporal windows ($seq\_len=8$).
7. **Lead-Time Analysis**: Evaluating skill at 10-day, 20-day, and 30-day forecast horizons.
8. **Held-Out Test Set Benchmark**: Comparative evaluation and zero-leakage verification.
9. **Export Bundle**: Compresses trained models (`.pkl`, `.json`, `.pth`) and reports into `trained_models.zip` and triggers an automatic download.

---

## 🛠️ Step-by-Step Instructions for Google Colab

### Option A: Upload to Colab Directly (Fastest)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **Upload** and select [`notebooks/train_all_models_colab.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/train_all_models_colab.ipynb).
3. Enable GPU Acceleration:
   - Click `Runtime` in the top menu $\to$ `Change runtime type`.
   - Under **Hardware accelerator**, select **T4 GPU**.
   - Click **Save**.
4. Upload Project Files (Optional but recommended):
   - In Colab's left sidebar, click the **Folder 📁** icon.
   - You can upload the `data/` folder directly, or let the notebook automatically generate the complete pilot dataset on the fly!
5. Click **Runtime** $\to$ **Run all** (`Ctrl + F9`).
6. When training finishes, Colab will automatically download `trained_models.zip` containing:
   - `models/rf_model.pkl`
   - `models/scaler.pkl`
   - `models/xgb_model.json`
   - `models/lstm_model.pth`
   - `reports/final_comparison.json`
   - `reports/lead_time_skill.png`
   - `reports/shap_summary.png`
   - `reports/rf_feature_importance.png`

---

### Option B: Using Google Drive

1. Upload the entire `venus` folder to your Google Drive under:
   `My Drive / FromSpaceToAction` or `My Drive / venus`
2. Open `notebooks/train_all_models_colab.ipynb` with Google Colab.
3. Select **T4 GPU** runtime.
4. Run all cells: the notebook will mount Google Drive, recognize your files, save models directly into your Drive's `models/` directory, and output all reports.

---

## 🔬 Modular Experimentation Notebooks

If you prefer to train and analyze models individually, use the modular notebooks:
- [`notebooks/06_baseline_rule_model.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/06_baseline_rule_model.ipynb) — Operational VCI baseline
- [`notebooks/07_random_forest.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/07_random_forest.ipynb) — Random Forest classifier & feature ranking
- [`notebooks/08_xgboost.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/08_xgboost.ipynb) — XGBoost with Modality Ablation & SHAP
- [`notebooks/09_lstm_forecasting.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/09_lstm_forecasting.ipynb) — PyTorch recurrent sequence forecasting on GPU
- [`notebooks/10_model_comparison.ipynb`](file:///c:/Users/kalwo/Desktop/venus/notebooks/10_model_comparison.ipynb) — Held-out comparison & leakage verification

---

## 📊 Integrating Trained Models with the Local Dashboard

Once you download `trained_models.zip` from Colab:
1. Extract the contents into your project:
   - Put model files into [`models/`](file:///c:/Users/kalwo/Desktop/venus/models/)
   - Put reports into [`outputs/reports/`](file:///c:/Users/kalwo/Desktop/venus/outputs/reports/)
2. Start the local Flask web dashboard:
   ```bash
   python dashboard/app.py
   ```
3. Open `http://localhost:5000/models` in your browser.
4. The dashboard will automatically detect `outputs/reports/final_comparison.json` and display your **real trained model metrics, lead-time curves, and risk scores**!
