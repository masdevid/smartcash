"""
File: smartcash/ui/cells/cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset
"""

from IPython.display import display
from smartcash.ui.dataset.download.download_initializer import initialize_dataset_download_ui

# Inisialisasi dan tampilkan UI
ui = initialize_dataset_download_ui()
display(ui)