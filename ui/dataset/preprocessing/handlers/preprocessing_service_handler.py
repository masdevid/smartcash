"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_service_handler.py
Deskripsi: Handler utama untuk preprocessing dataset (menggunakan SRP files)
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

# Import handler SRP files
from smartcash.ui.dataset.preprocessing.handlers.service_handler import get_dataset_manager, run_preprocessing
from smartcash.ui.dataset.preprocessing.handlers.parameter_handler import extract_preprocess_params, validate_preprocessing_params
from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import ensure_ui_persistence, get_preprocessing_config, sync_config_with_drive

logger = get_logger("preprocessing_handler")

def setup_preprocessing_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup semua handler untuk preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Pastikan persistensi UI components
        ui_components = ensure_ui_persistence(ui_components)
        
        # Setup button handlers
        from smartcash.ui.dataset.preprocessing.handlers.button_handlers import setup_button_handlers
        ui_components = setup_button_handlers(ui_components)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat setup preprocessing handlers: {str(e)}")
        return ui_components

def run_preprocessing_with_ui(ui_components: Dict[str, Any]) -> bool:
    """
    Jalankan preprocessing dataset dengan UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan konfigurasi preprocessing dari UI
        params = get_preprocessing_config(ui_components)
        
        # Jalankan preprocessing
        success = run_preprocessing(ui_components, params)
        
        # Update status
        status_type = 'success' if success else 'error'
        message = f"{ICONS['success' if success else 'error']} Preprocessing {'berhasil' if success else 'gagal'} dijalankan"
        
        # Update status panel
        from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
        update_status_panel(ui_components, status_type, message)
        
        # Sinkronkan konfigurasi dengan drive
        sync_config_with_drive(ui_components)
        
        return success
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat menjalankan preprocessing: {str(e)}")
        return False

def initialize_preprocessing(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inisialisasi preprocessing dengan persistensi konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Pastikan persistensi UI components
        ui_components = ensure_ui_persistence(ui_components)
        
        # Setup handlers
        ui_components = setup_preprocessing_handlers(ui_components)
        
        # Deteksi state preprocessing
        from smartcash.ui.dataset.preprocessing.handlers.state_handler import detect_preprocessing_state
        ui_components = detect_preprocessing_state(ui_components)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat inisialisasi preprocessing: {str(e)}")
        return ui_components

# Fungsi utilitas untuk membantu integrasi dengan modul lain
def get_preprocessing_manager(ui_components: Dict[str, Any]) -> Any:
    """
    Dapatkan preprocessing manager untuk integrasi dengan modul lain.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance preprocessing manager
    """
    return get_dataset_manager(ui_components)
