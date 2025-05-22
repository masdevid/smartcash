"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Cell untuk augmentasi dataset mengikuti pola dataset download
"""

from IPython.display import display
from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_dataset_augmentation_ui

# Inisialisasi dan tampilkan UI
ui = initialize_dataset_augmentation_ui()
display(ui)