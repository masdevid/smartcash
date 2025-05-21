"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Initializer minimalis untuk modul preprocessing dataset dengan pendekatan DRY
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

# Import core functionality
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    PREPROCESSING_LOGGER_NAMESPACE,
    get_observer_manager
)
from smartcash.ui.dataset.preprocessing.handlers.setup_handlers import setup_preprocessing_handlers
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.preprocessing.utils.ui_observers import register_ui_observers
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
from smartcash.ui.dataset.preprocessing.components.preprocessing_component import create_preprocessing_ui

# Singleton instance untuk UI components
_INITIALIZED_COMPONENTS = None

def initialize_preprocessing_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk preprocessing dataset.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi aplikasi (optional)
        
    Returns:
        Widget UI utama yang bisa ditampilkan
    """
    global _INITIALIZED_COMPONENTS
    
    # Setup logger
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Jika sudah diinisialisasi, kembalikan UI yang ada
    if _INITIALIZED_COMPONENTS and 'ui' in _INITIALIZED_COMPONENTS:
        logger.debug(f"{ICONS['info']} UI preprocessing dataset sudah diinisialisasi sebelumnya")
        return _INITIALIZED_COMPONENTS['ui']
    
    logger.info(f"{ICONS['start']} Memulai inisialisasi UI preprocessing dataset")
    
    try:
        # Load config
        config_manager = get_config_manager()
        dataset_config = config_manager.get_module_config('dataset') or {}
        
        # Merge dengan config yang diberikan
        if config:
            _merge_configs(dataset_config, config)
        
        # Buat UI components
        ui_components = create_preprocessing_ui(dataset_config)
        
        # Setup logger
        ui_components = setup_ui_logger(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
        
        # Setup observer sistem dan flags
        ui_components = _setup_core_components(ui_components)
        
        # Setup semua handlers (termasuk handlers untuk config, button, status, dll)
        ui_components = setup_preprocessing_handlers(ui_components, env, dataset_config)
        
        # Set status awal
        update_status_panel(ui_components, "idle", "Preprocessing siap dijalankan")
        
        # Set singleton
        _INITIALIZED_COMPONENTS = ui_components
        
        # Log success
        log_message(ui_components, "UI preprocessing dataset berhasil diinisialisasi", "success", "✅")
        
        # Return main UI untuk ditampilkan
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat inisialisasi UI preprocessing: {str(e)}")
        return _create_error_ui(str(e))

def get_preprocessing_ui_components() -> Optional[Dict[str, Any]]:
    """
    Dapatkan UI components yang sudah diinisialisasi.
    
    Returns:
        Dictionary komponen UI atau None jika belum diinisialisasi
    """
    global _INITIALIZED_COMPONENTS
    return _INITIALIZED_COMPONENTS

def reset_preprocessing_ui(env=None, config=None) -> Any:
    """
    Reset UI preprocessing dataset ke kondisi awal.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Widget UI utama yang baru
    """
    global _INITIALIZED_COMPONENTS
    
    # Reset singleton
    _INITIALIZED_COMPONENTS = None
    
    # Logger
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    logger.info(f"{ICONS['reset']} Mereset UI preprocessing dataset")
    
    # Inisialisasi ulang
    return initialize_preprocessing_ui(env, config)

def _setup_core_components(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup komponen inti seperti observer, flags, dll.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang diupdate
    """
    # Tambahkan flag untuk tracking status
    ui_components['preprocessing_initialized'] = True
    ui_components['preprocessing_running'] = False
    
    # Setup observer
    observer_manager = get_observer_manager()
    ui_components['observer_manager'] = observer_manager
    ui_components['observer_group'] = "preprocessing_observers"
    
    # Register UI observers
    register_ui_observers(ui_components)
    
    return ui_components

def _merge_configs(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration secara rekursif dengan smart merging.
    
    Args:
        base_config: Konfigurasi dasar yang akan diupdate
        update_config: Konfigurasi baru untuk mengupdate base
        
    Returns:
        Konfigurasi yang sudah diupdate
    """
    # Iterate melalui semua keys di update_config
    for key, value in update_config.items():
        # Nested dict handling
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    
    return base_config

def _create_error_ui(error_message: str) -> widgets.VBox:
    """
    Buat UI error jika inisialisasi gagal.
    
    Args:
        error_message: Pesan error
        
    Returns:
        Widget UI error
    """
    try:
        from smartcash.ui.utils.alert_utils import create_error_alert
        
        error_alert = create_error_alert(
            error_message,
            "Error Inisialisasi Preprocessing"
        )
        
        return widgets.VBox([
            widgets.HTML(f"<h2>{ICONS['error']} Error Inisialisasi UI Preprocessing</h2>"),
            error_alert
        ])
    except Exception:
        # Fallback sederhana jika create_error_alert gagal diimpor
        return widgets.HTML(
            f"<div style='color:red; padding:10px; border:1px solid red;'>"
            f"<h3>{ICONS.get('error', '❌')} Error Inisialisasi UI Preprocessing</h3>"
            f"<p>{error_message}</p></div>"
        )