"""
File: smartcash/ui/cells/cell_3_3_training_strategy.py
Deskripsi: Entry point untuk konfigurasi strategi training model SmartCash
"""

def setup_training_strategy():
    """Setup dan tampilkan UI untuk konfigurasi strategi training."""
    # Import modul training strategy
    from smartcash.ui.training_config.training_strategy.training_strategy_initializer import initialize_training_strategy_ui
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_training_strategy_ui()

# Eksekusi saat modul diimpor
ui_components = setup_training_strategy()