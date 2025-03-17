"""
File: smartcash/ui/setup/cell_1_1_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash
"""

import sys
import os
from pathlib import Path

# Pastikan direktori project ada di path
if '.' not in sys.path:
    sys.path.append('.')

# Setup environment
from smartcash.ui.components.cell_template import run_cell

def main():
    """
    Konfigurasi environment utama untuk SmartCash.
    
    Returns:
        Dictionary komponen UI untuk konfigurasi environment
    """
    # Path konfigurasi default
    config_path = "configs/colab_config.yaml"
    
    # Jalankan cell dengan konfigurasi environment
    ui_components = run_cell("env_config", config_path)
    
    return ui_components

# Jalankan konfigurasi
if __name__ == "__main__":
    main()