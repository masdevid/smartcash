"""
File: smartcash/ui/cells/cell_2_2_preprocessing.py
Deskripsi: Entry point untuk cell preprocessing dataset
"""

def setup_preprocessing():
    """Setup dan tampilkan UI untuk preprocessing dataset."""
    from smartcash.ui.dataset.preprocessing.preprocessing_initializer import initialize_preprocessing_ui
    return initialize_preprocessing_ui()

# Eksekusi saat modul diimpor
ui_components = setup_preprocessing()