"""
File: smartcash/ui/cells/cell_3_2_hyperparameters.py
Deskripsi: Entry point untuk konfigurasi hyperparameter model SmartCash
"""

from smartcash.ui.training_config.hyperparameters.hyperparameters_initializer import (
    get_hyperparameters_ui,
    display_hyperparameters_ui
)

# Inisialisasi dan tampilkan UI hyperparameter
ui_components = get_hyperparameters_ui()
display_hyperparameters_ui(ui_components)