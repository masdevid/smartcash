"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Cell untuk augmentasi dataset mengikuti pola dataset download
"""

from IPython.display import display
from smartcash.ui.dataset.augmentation.augmentation_initializer import init_augmentation

# Inisialisasi dan tampilkan UI
ui = init_augmentation()
display(ui)