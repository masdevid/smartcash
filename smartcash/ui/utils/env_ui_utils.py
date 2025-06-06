"""
File: smartcash/ui/utils/env_ui_utils.py
Deskripsi: Utilitas UI untuk environment config dengan namespace yang sudah diperbaiki
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
import sys
from datetime import datetime
from IPython.display import display, HTML

from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger import log_to_ui

# Konstanta namespace yang diperbaiki berdasarkan KNOWN_NAMESPACES
ENV_CONFIG_LOGGER_NAMESPACE = "smartcash.ui.env_config"
MODULE_LOGGER_NAME = "ENV"

def update_status(ui_components: Dict[str, Any], message: str, style: str = "info") -> None:
    """
    Update status panel with alert
    
    Args:
        ui_components: Dictionary UI components
        message: Status message
        style: Alert style (info, success, error)
    """
    from smartcash.ui.utils.alert_utils import create_info_box
    ui_components['status_panel'].value = create_info_box(
        "Environment Status",
        message,
        style=style
    ).value

def set_button_state(button: widgets.Button, disabled: bool, style: str = None) -> None:
    """
    Update button state
    
    Args:
        button: Button widget
        disabled: Whether button should be disabled
        style: Button style
    """
    button.disabled = disabled
    if style:
        button.button_style = style

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log message ke UI dan logger Python dengan namespace environment config.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional untuk ditampilkan di depan pesan
    """
    # Jika pesan kosong, jangan log
    if not message or not message.strip():
        return
        
    # Pastikan komponen UI memiliki namespace yang tepat
    if 'logger_namespace' not in ui_components:
        ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
    
    # Tandai sebagai modul environment config
    if 'env_config_initialized' not in ui_components:
        ui_components['env_config_initialized'] = True
    
    # Gunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(ENV_CONFIG_LOGGER_NAMESPACE)
    
    # Log ke UI dengan log_to_ui untuk format yang konsisten
    log_to_ui(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering di logger Python
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
    
    # Log ke Python logger juga
    if logger:
        if level == "info":
            logger.info(prefixed_message)
        elif level == "warning" or level == "warn":
            logger.warning(prefixed_message)
        elif level == "error":
            logger.error(prefixed_message)
        elif level == "debug":
            logger.debug(prefixed_message)
        elif level == "success":
            # Success level tidak ada di Python logger standard, gunakan info
            logger.info(f"✅ {prefixed_message}")
        elif level == "critical":
            logger.critical(prefixed_message)

def update_progress(ui_components: Dict[str, Any], value: float, message: str = "") -> None:
    """
    Update progress bar and message
    
    Args:
        ui_components: Dictionary UI components
        value: Progress value (0-1)
        message: Progress message
    """
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
    
    if 'progress_message' in ui_components and message:  # Hanya update jika message tidak kosong
        ui_components['progress_message'].value = message
        
    # Log progress message jika ada
    if message:
        log_message(ui_components, message, "info")