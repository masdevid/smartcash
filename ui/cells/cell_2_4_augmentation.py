"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Entry point untuk cell augmentasi dataset
"""

def setup_augmentation():
    """Setup dan tampilkan UI untuk augmentasi dataset."""
    from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_augmentation_ui
    return initialize_augmentation_ui()

# Eksekusi saat modul diimpor
ui_components = setup_augmentation()