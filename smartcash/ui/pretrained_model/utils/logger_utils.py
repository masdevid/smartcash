"""
File: smartcash/ui/pretrained_model/utils/logger_utils.py
Deskripsi: Utilitas untuk logging yang konsisten di modul pretrained model
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import (
    PRETRAINED_MODEL_LOGGER_NAMESPACE, 
    KNOWN_NAMESPACES,
    create_namespace_badge
)

# Konstanta untuk namespace logger
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PRETRAINED_MODEL_LOGGER_NAMESPACE]
logger = get_logger(PRETRAINED_MODEL_LOGGER_NAMESPACE)

def get_module_logger():
    """
    Mendapatkan logger yang sudah dikonfigurasi dengan namespace yang benar
    """
    return logger

def format_log_message(message: str, level: str = "info") -> str:
    """
    Format pesan log dengan styling yang konsisten
    
    Args:
        message: Pesan yang akan diformat
        level: Level log (info, success, warning, error)
        
    Returns:
        Pesan yang sudah diformat
    """
    level_icon = {
        "info": ICONS.get('info', 'ℹ️'),
        "success": ICONS.get('success', '✅'),
        "warning": ICONS.get('warning', '⚠️'),
        "error": ICONS.get('error', '❌')
    }.get(level, 'ℹ️')
    
    return f"{level_icon} {message}"

def format_log_message_html(message: str, level: str = "info") -> str:
    """
    Format pesan log dengan styling HTML yang konsisten
    
    Args:
        message: Pesan yang akan diformat
        level: Level log (info, success, warning, error)
        
    Returns:
        HTML string yang sudah diformat
    """
    namespace_badge = create_namespace_badge("PRETRAIN")
    
    level_icon = {
        "info": ICONS.get('info', 'ℹ️'),
        "success": ICONS.get('success', '✅'),
        "warning": ICONS.get('warning', '⚠️'),
        "error": ICONS.get('error', '❌')
    }.get(level, 'ℹ️')
    
    return f"""
    <div style="margin:2px 0;padding:4px 8px;border-radius:4px;
               background-color:rgba(248,249,250,0.8);
               border-left:3px solid #D7BDE2;">
        <span style="margin-right:5px;">{namespace_badge}</span>
        <span>{level_icon} {message}</span>
    </div>
    """

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """
    Log pesan ke komponen UI dengan styling yang konsisten
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan dilog
        level: Level log (info, success, warning, error)
    """
    # Log ke UI jika tersedia
    if 'log' in ui_components and hasattr(ui_components['log'], 'append_display_data'):
        html_message = format_log_message_html(message, level)
        with ui_components['log']:
            display(HTML(html_message))
    
    # Log ke logger
    if level == "info":
        logger.info(message)
    elif level == "success":
        logger.info(format_log_message(message, "success"))
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)
