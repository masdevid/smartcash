"""
File: smartcash/ui/cells/cell_3_1_backbone_selection.py
Deskripsi: Cell untuk pemilihan model, backbone dan konfigurasi layer SmartCash yang kompatibel dengan ModelManager
"""

# Import dan jalankan skeleton cell dengan parameter yang sesuai
from smartcash.ui.training_config.cell_skeleton import run_cell

# Jalankan cell dengan parameter konfigurasi
ui_components = run_cell("backbone_selection", "configs/model_config.yaml")