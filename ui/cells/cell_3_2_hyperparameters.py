"""
File: smartcash/ui/cells/cell_3_2_hyperparameters.py
Deskripsi: Entry point untuk konfigurasi hyperparameter model SmartCash
"""

def setup_hyperparameters():
    """Setup dan tampilkan UI untuk konfigurasi hyperparameter model."""
    # Import modul hyperparameters
    from smartcash.ui.training_config.hyperparameters.hyperparameters_initializer import initialize_hyperparameters_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_hyperparameters_ui()

# Eksekusi saat modul diimpor
ui_components = setup_hyperparameters()