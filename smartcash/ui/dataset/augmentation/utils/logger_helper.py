"""
File: smartcash/ui/dataset/augmentation/utils/logger_helper.py
Deskripsi: Helper untuk logging UI augmentasi yang konsisten
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Import namespace konstanta
from smartcash.ui.dataset.augmentation.augmentation_initializer import AUGMENTATION_LOGGER_NAMESPACE, MODULE_LOGGER_NAME

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan namespace khusus augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional
    """
    # Cek apakah ini adalah augmentasi yang sudah diinisialisasi
    if not is_initialized(ui_components):
        return
    
    # Pastikan menggunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
        ui_log(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
    
    # Log ke Python logger
    if logger:
        getattr(logger, level if level != "success" else "info")(
            f"✅ {prefixed_message}" if level == "success" else prefixed_message
        )

def setup_ui_logger(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup UI logger untuk module augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah ditambahkan logger
    """
    # Setup logger jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Tambahkan fungsi log_message ke UI components
    ui_components['log_message'] = lambda msg, level="info", icon=None: log_message(ui_components, msg, level, icon)
    
    # Tambahkan flag augmentation_initialized
    ui_components['augmentation_initialized'] = True
    
    # Tambahkan namespace
    ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
    
    return ui_components 

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI logger sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika sudah diinisialisasi
    """
    return ui_components.get('augmentation_initialized', False)
