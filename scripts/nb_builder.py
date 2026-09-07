# From Space to Action — Notebook Generation Utility
"""
Creates .ipynb (Jupyter/Colab notebook) files from a simple Python specification.
Each notebook is defined by a list of cells (markdown or code).

Usage:
    from nb_builder import NotebookBuilder
    nb = NotebookBuilder("My Notebook")
    nb.md("# Title\nDescription here")
    nb.code("print('hello')")
    nb.save("output.ipynb")
"""

import json
import os

class NotebookBuilder:
    def __init__(self, title="Untitled", use_gpu=False):
        self.title = title
        self.use_gpu = use_gpu
        self.cells = []
    
    def md(self, source):
        """Add a markdown cell."""
        lines = source.split('\n')
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]]
        self.cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source_lines
        })
        return self
    
    def code(self, source, collapsed=False):
        """Add a code cell."""
        lines = source.split('\n')
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]]
        metadata = {}
        if collapsed:
            metadata["cellView"] = "form"
        self.cells.append({
            "cell_type": "code",
            "metadata": metadata,
            "source": source_lines,
            "execution_count": None,
            "outputs": []
        })
        return self
    
    def build(self):
        """Build the notebook JSON structure."""
        metadata = {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        }
        if self.use_gpu:
            metadata["accelerator"] = "GPU"
            metadata["colab"]["gpuType"] = "T4"
        
        return {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": metadata,
            "cells": self.cells
        }
    
    def save(self, filepath):
        """Save notebook to .ipynb file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        nb = self.build()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Created: {filepath} ({len(self.cells)} cells)")
        return filepath


# Standard Colab header for all notebooks
COLAB_HEADER = '''# === Google Colab Setup ===
import os, sys

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Project paths
PROJECT_DIR = '/content/drive/MyDrive/FromSpaceToAction'
DATA_DIR = f'{PROJECT_DIR}/data'
SRC_DIR = f'{PROJECT_DIR}/src'
MODELS_DIR = f'{PROJECT_DIR}/models'
OUTPUTS_DIR = f'{PROJECT_DIR}/outputs'
CONFIG_PATH = f'{PROJECT_DIR}/config/config.yaml'

# Create directories
for d in [DATA_DIR, SRC_DIR, MODELS_DIR, OUTPUTS_DIR,
          f'{DATA_DIR}/raw', f'{DATA_DIR}/processed',
          f'{DATA_DIR}/features', f'{DATA_DIR}/targets',
          f'{OUTPUTS_DIR}/maps', f'{OUTPUTS_DIR}/reports']:
    os.makedirs(d, exist_ok=True)

# Add src to path
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_DIR)'''

COLAB_INSTALL_GEO = '''# Install geospatial packages (takes ~1 min)
!pip install -q earthengine-api geemap rasterio geopandas folium plotly pyyaml xgboost shap tqdm'''

COLAB_INSTALL_ML = '''# Install ML packages
!pip install -q xgboost shap'''

GEE_AUTH = '''# Authenticate & Initialize Google Earth Engine
import ee

try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE already authenticated")
except Exception:
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE authenticated and initialized")'''

LOAD_CONFIG = '''# Load project configuration
import yaml

with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

PILOT_BBOX = CONFIG['study_area']['pilot_bbox']
DATE_START = CONFIG['dates']['start']
DATE_END = CONFIG['dates']['end']
CLIM_START = CONFIG['dates']['climatology_start']

print(f"📍 Study area: {CONFIG['study_area']['name']}")
print(f"📅 Period: {DATE_START} to {DATE_END}")
print(f"📦 Pilot bbox: {PILOT_BBOX}")'''


if __name__ == '__main__':
    # Quick test
    nb = NotebookBuilder("Test Notebook")
    nb.md("# Test\nThis is a test notebook.")
    nb.code("print('Hello from Colab!')")
    nb.save("test_output.ipynb")
    print("Builder works!")
