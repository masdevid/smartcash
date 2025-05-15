"""
File: smartcash/ui/cells/cell_2_5_dataset_visualization.py
Deskripsi: Entry point untuk visualisasi dataset dengan pendekatan DRY
"""

def setup_dataset_visualization():
    """Setup dan tampilkan UI untuk visualisasi dataset."""
    # Import modul visualisasi dataset
    from smartcash.ui.dataset.visualization.visualization_initializer import initialize_visualization_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_visualization_ui()

# Eksekusi saat modul diimpor
ui_components = setup_dataset_visualization()
