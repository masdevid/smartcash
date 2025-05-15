"""
File: smartcash/ui/cells/cell_2_3_preprocessing.py
Deskripsi: Entry point untuk preprocessing dataset dengan pendekatan DRY
"""

def setup_preprocessing():
    """Setup dan tampilkan UI untuk preprocessing dataset."""
    # Import modul preprocessing
    from smartcash.ui.dataset.preprocessing.preprocessing_initializer import initialize_preprocessing_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_preprocessing_ui()

# Eksekusi saat modul diimpor
ui_components = setup_preprocessing()