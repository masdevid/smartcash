"""
File: smartcash/ui/cells/cell_2_1_dataset_downloader.py
Deskripsi: Entry point untuk dataset downloader cell
"""

def setup_dataset_downloader():
    """Setup dan tampilkan UI untuk download dataset."""
    from smartcash.ui.dataset.dataset_downloader_initializer import initialize_dataset_downloader
    return initialize_dataset_downloader()

# Eksekusi saat modul diimpor
ui_components = setup_dataset_downloader()