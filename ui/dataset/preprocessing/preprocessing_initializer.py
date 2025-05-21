"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer untuk UI preprocessing dataset dengan perbaikan tombol stop
"""

from typing import Dict, Any, Optional
# Path tidak digunakan langsung tetapi mungkin dibutuhkan untuk type hints di komponen lain
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger

# Konstanta untuk namespace logger
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"
# Konstanta untuk ID namespace di UI
MODULE_LOGGER_NAME = "PREPROCESSING"

# Import utils yang dibutuhkan untuk inisialisasi
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, is_initialized
from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel, reset_ui_after_preprocessing
from smartcash.ui.dataset.preprocessing.utils.progress_manager import reset_progress_bar

# Import setup handlers
from smartcash.ui.dataset.preprocessing.handlers.setup_handlers import setup_preprocessing_handlers
from smartcash.ui.dataset.preprocessing.components.ui_factory import create_preprocessing_ui_components

# Flag global untuk mencegah inisialisasi ulang
_PREPROCESSING_MODULE_INITIALIZED = False

def initialize_dataset_preprocessing_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset preprocessing.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _PREPROCESSING_MODULE_INITIALIZED
    
    # Setup logger dengan nama modul spesifik
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Hindari multiple inisialisasi yang tidak perlu
    if _PREPROCESSING_MODULE_INITIALIZED:
        logger.debug(f"[{MODULE_LOGGER_NAME}] UI preprocessing dataset sudah diinisialisasi sebelumnya, menggunakan inisialisasi yang sudah ada")
    else:
        logger.info(f"[{MODULE_LOGGER_NAME}] Memulai inisialisasi UI preprocessing dataset")
        _PREPROCESSING_MODULE_INITIALIZED = True
    
    try:
        # Get config manager dengan fallback otomatis
        config_manager = get_config_manager()
        
        # Get dataset config dari SimpleConfigManager
        dataset_config = config_manager.get_module_config('dataset')
        
        # Merge dengan config yang diberikan
        if config:
            dataset_config.update(config)
            
        # Create UI components
        ui_components = create_preprocessing_ui_components(dataset_config)
        
        # Setup logger dengan namespace spesifik
        from smartcash.ui.dataset.preprocessing.utils.logger_helper import setup_ui_logger
        ui_components = setup_ui_logger(ui_components)
        ui_components['preprocessing_initialized'] = True
        
        # Tambahkan flag untuk tracking status
        ui_components['preprocessing_running'] = False
        ui_components['cleanup_running'] = False
        ui_components['stop_requested'] = False
        
        # Pastikan tombol stop tersembunyi di awal
        if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
            ui_components['stop_button'].layout.display = 'none'
        
        # Setup handlers
        ui_components = setup_preprocessing_handlers(ui_components, env, dataset_config)
        
        # Log bahwa inisialisasi berhasil
        from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
        log_message(ui_components, "UI preprocessing dataset berhasil diinisialisasi", "info", "✅")
        
        # Return main UI widget
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"[{MODULE_LOGGER_NAME}] Error saat inisialisasi UI preprocessing: {str(e)}")
        # Tampilkan error UI sederhana
        import ipywidgets as widgets
        from smartcash.ui.utils.constants import ICONS, COLORS
        
        error_ui = widgets.VBox([
            widgets.HTML(f"<h2 style='color: {COLORS['danger']};'>{ICONS['error']} Error Inisialisasi Preprocessing</h2>"),
            widgets.HTML(f"<div style='color: {COLORS['danger']};'>{str(e)}</div>"),
            widgets.HTML(f"<div>Silakan periksa log atau hubungi administrator.</div>")
        ])
        
        return error_ui

# Alias untuk kompatibilitas
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui