"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Entry point untuk augmentasi dataset dengan pendekatan DRY
"""

def setup_augmentation():
    """Setup dan tampilkan UI untuk augmentasi dataset."""
    # Import modul augmentation
    from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_augmentation_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_augmentation_ui()

# Eksekusi saat modul diimpor
ui_components = setup_augmentation()