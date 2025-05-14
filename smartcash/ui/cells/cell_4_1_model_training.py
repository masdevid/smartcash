"""
File: smartcash/ui/cells/cell_4_1_model_training.py
Deskripsi: Entry point untuk proses training model SmartCash
"""

def setup_model_training():
    """Setup dan tampilkan UI untuk proses training model."""
    # Import modul training
    from smartcash.ui.training.training_initializer import initialize_training_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_training_ui()

# Eksekusi saat modul diimpor
ui_components = setup_model_training()
