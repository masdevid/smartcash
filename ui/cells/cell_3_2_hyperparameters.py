"""
File: smartcash/ui/cells/cell_3_2_hyperparameters.py
Deskripsi: Cell untuk konfigurasi hyperparameter training model SmartCash dengan memanfaatkan skeleton
"""

# Import dan jalankan skeleton cell dengan parameter yang sesuai
from smartcash.ui.training_config.cell_skeleton import run_cell

# Jalankan cell dengan parameter konfigurasi
ui_components = run_cell("hyperparameters", "configs/training_config.yaml")