"""
File: smartcash/ui/dataset/preprocessing/utils/logger_helper.py
Deskripsi: Utilitas untuk membantu logging di UI preprocessing
"""

from typing import Dict, Any, Optional, Callable
import logging
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import create_ui_logger

# Konstanta untuk namespace logger
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"

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
        
        # Setup UI logger
        if 'log_output' in ui_components:
            ui_logger = create_ui_logger(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
            ui_components['ui_logger'] = ui_logger
            
        # Setup log_message function
        ui_components['log_message'] = lambda msg, level='info', icon='': log_message(ui_components, msg, level, icon)
        
    return ui_components

def log_message(ui_components: Dict[str, Any], message: str, level: str = 'info', icon: str = '') -> None:
    """
    Log pesan dengan level yang diberikan.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan dilog
        level: Level log (debug, info, warning, error, critical)
        icon: Ikon untuk pesan (emoji)
    """
    logger = ui_components.get('logger')
    if not logger:
        # Fallback ke logger baru
        logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
        ui_components['logger'] = logger
    
    # Format pesan dengan icon
    formatted_message = f"{icon} {message}" if icon else message
    
    # Log ke logger
    if level == 'debug':
        logger.debug(formatted_message)
    elif level == 'info':
        logger.info(formatted_message)
    elif level == 'warning':
        logger.warning(formatted_message)
    elif level == 'error':
        logger.error(formatted_message)
    elif level == 'critical':
        logger.critical(formatted_message)
    else:
        logger.info(formatted_message)
    
    # Log ke UI widget jika ada
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
        ui_components['log_output'].append_log(formatted_message, level)

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI preprocessing sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean menunjukkan apakah UI sudah diinisialisasi
    """
    return ui_components.get('preprocessing_initialized', False) 