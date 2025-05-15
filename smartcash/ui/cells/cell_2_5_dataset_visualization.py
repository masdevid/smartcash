"""
File: smartcash/ui/cells/cell_2_5_dataset_visualization.py
Deskripsi: Entry point minimalis untuk visualisasi dataset
"""

import sys
import os

# Pastikan path modul dapat diakses
try:
    # Untuk lingkungan normal (non-Colab)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    # Untuk lingkungan Colab
    sys.path.append('/content/smartcash')

# Import fungsi setup yang telah dipisahkan
from smartcash.ui.dataset.visualization.setup import setup_dataset_visualization, reset_visualization, is_restart_mode

# Eksekusi saat modul diimpor
if __name__ == "__main__":
    # Jika dalam mode restart, paksa reset visualisasi
    if is_restart_mode():
        ui_components = reset_visualization()
    else:
        ui_components = setup_dataset_visualization()
else:
    # Jika diimpor dari modul lain, gunakan setup normal
    ui_components = setup_dataset_visualization()
