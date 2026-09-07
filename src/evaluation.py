"""
evaluation.py

Metrics and reporting routines for model performance.
"""

import pandas as pd
import numpy as np

try:
    from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    pass

def compute_classification_metrics(y_true, y_pred, class_names):
    """Computes precision, recall, F1 per class + macro."""
    return classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

def compute_far(y_true, y_pred, positive_classes: list = [1, 2, 3]):
    """False Alarm Ratio."""
    true_pos = sum((y_pred == p) & (y_true == p) for p in positive_classes)
    false_pos = sum((y_pred == p) & (y_true != p) for p in positive_classes)
    if true_pos + false_pos == 0: return 0.0
    return false_pos / (true_pos + false_pos)

def compute_pod(y_true, y_pred, positive_classes: list = [1, 2, 3]):
    """Probability of Detection (Recall)."""
    true_pos = sum((y_pred == p) & (y_true == p) for p in positive_classes)
    actual_pos = sum(y_true == p for p in positive_classes)
    if actual_pos == 0: return 0.0
    return true_pos / actual_pos

def compute_regression_metrics(y_true, y_pred):
    """RMSE, MAE, R-squared."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def lead_time_analysis(model, X, y, leads, **kwargs):
    """Analyzes model skill vs lead time."""
    return {}

def ablation_report(results_dict: dict) -> pd.DataFrame:
    """Formats ablation study results as DataFrame."""
    return pd.DataFrame.from_dict(results_dict, orient='index')

def generate_model_comparison_table(results: list) -> pd.DataFrame:
    """Formatted comparison table for different models."""
    return pd.DataFrame(results)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Matplotlib confusion matrix plotting placeholder."""
    pass
