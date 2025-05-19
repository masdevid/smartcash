"""
File: smartcash/ui/cells/cell_4_1_model_training.py
Deskripsi: Entry point untuk proses training model SmartCash
"""

def setup_model_training():
    """Setup dan tampilkan UI untuk proses training model."""
    # Import modul training dengan komponen baru
    from smartcash.ui.training.training_initializer import initialize_training_ui
    from smartcash.common.environment import get_environment_manager
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan environment manager dan config manager
    env = get_environment_manager()
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi yang sudah ada
    model_config = config_manager.get_module_config('model', {})
    hyperparameters_config = config_manager.get_module_config('hyperparameters', {})
    training_strategy_config = config_manager.get_module_config('training_strategy', {})
    
    # Gabungkan konfigurasi
    combined_config = {
        'model': model_config,
        'hyperparameters': hyperparameters_config,
        'training_strategy': training_strategy_config
    }
    
    # Inisialisasi UI dan kembalikan komponen
    return initialize_training_ui(env, combined_config)

# Eksekusi saat modul diimpor
ui_components = setup_model_training()
