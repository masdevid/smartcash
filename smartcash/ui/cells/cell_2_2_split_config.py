"""
File: smartcash/ui/cells/cell_2_2_split_config.py
Deskripsi: Entry point untuk konfigurasi split dataset
"""

def setup_split_config():
    """Setup dan tampilkan UI untuk konfigurasi split dataset."""
    from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
    return initialize_split_ui()

# Eksekusi saat modul diimpor
ui_components = setup_split_config()
