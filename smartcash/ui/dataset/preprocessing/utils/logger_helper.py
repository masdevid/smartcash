"""
File: smartcash/ui/dataset/preprocessing/utils/logger_helper.py
Deskripsi: Utilitas untuk membantu logging di UI preprocessing
"""

from typing import Dict, Any, Optional, Callable
import logging
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger, log_to_ui as ui_log

# Import namespace konstanta
from smartcash.ui.dataset.preprocessing.preprocessing_initializer import PREPROCESSING_LOGGER_NAMESPACE, MODULE_LOGGER_NAME

def setup_ui_logger(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup logger untuk UI preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    if not ui_components.get('logger'):
        # Setup logger dengan namespace yang sudah didefine
        logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
        ui_components['logger'] = logger
        ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
    
        # Setup UI logger dengan namespace dan display name spesifik
        if 'log_output' in ui_components:
            ui_logger = create_ui_logger(
                ui_components, 
                PREPROCESSING_LOGGER_NAMESPACE,
                redirect_stdout=False
            )
            ui_components['ui_logger'] = ui_logger
            
        # Setup log_message function
        ui_components['log_message'] = lambda msg, level='info', icon='': log_message(ui_components, msg, level, icon)
        
        # Tandai modul sudah diinisialisasi
        ui_components['preprocessing_initialized'] = True
    
    return ui_components

def log_message(ui_components: Dict[str, Any], message: str, level: str = 'info', icon: str = '') -> None:
    """
    Log pesan ke UI dan logger Python dengan namespace khusus preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (debug, info, warning, error, critical)
        icon: Ikon untuk pesan (emoji)
    """
    # Cek apakah ini adalah preprocessing yang sudah diinisialisasi
    if not is_initialized(ui_components):
        # Skip UI logging jika belum diinisialisasi untuk mencegah log muncul di modul lain
        return
    
    # Pastikan menggunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    # Format pesan dengan icon jika ada
    formatted_message = f"{icon} {message}" if icon else message
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
        # Log ke UI dengan konsisten menggunakan UI logger
        ui_log(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {formatted_message}"
    
    # Log ke Python logger
    if logger:
        if level == 'debug':
            logger.debug(prefixed_message)
        elif level == 'info':
            logger.info(prefixed_message)
        elif level == 'warning' or level == 'warn':
            logger.warning(prefixed_message)
        elif level == 'error':
            logger.error(prefixed_message)
        elif level == 'success':
            # Success level tidak ada di Python logger standard, gunakan info
            logger.info(f"✅ {prefixed_message}")
        elif level == 'critical':
            logger.critical(prefixed_message)

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI preprocessing sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean menunjukkan apakah UI sudah diinisialisasi
    """
    return ui_components.get('preprocessing_initialized', False) 