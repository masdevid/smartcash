"""
File: smartcash/ui/cells/cell_2_1_dataset_download.py
Deskripsi: Entry point untuk cell download dataset
"""

def setup_dataset_download():
    """Setup dan tampilkan UI untuk download dataset."""
    from smartcash.ui.dataset.download.download_initializer import initialize_dataset_download_ui
    return initialize_dataset_download_ui()

# Eksekusi saat modul diimpor
ui_components = setup_dataset_download()