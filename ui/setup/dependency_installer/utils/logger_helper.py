"""
File: smartcash/ui/setup/dependency_installer/utils/logger_helper.py
Deskripsi: Fixed logger helper dengan import path yang benar
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Fixed import path
DEPENDENCY_INSTALLER_LOGGER_NAMESPACE = "smartcash.setup.dependency_installer"
MODULE_LOGGER_NAME = "DEPS"

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """Log pesan ke UI dan logger dengan namespace dependency installer"""
    # Check initialization
    if not is_initialized(ui_components):
        return
    
    # Get logger
    logger = ui_components.get('logger') or get_logger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    # Log ke UI
    if 'log_output' in ui_components or 'output' in ui_components or 'status' in ui_components:
        ui_log(ui_components, message, level, icon)
    
    # Log ke Python logger dengan prefix
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
    
    if logger:
        log_methods = {
            "info": logger.info,
            "warning": logger.warning,
            "warn": logger.warning,
            "error": logger.error,
            "debug": logger.debug,
            "success": lambda msg: logger.info(f"✅ {msg}"),
            "critical": logger.critical
        }
        log_method = log_methods.get(level, logger.info)
        log_method(prefixed_message)

def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """Check apakah dependency installer sudah diinisialisasi"""
    return ui_components.get('dependency_installer_initialized', False) or ui_components.get('module_name') == 'dependency_installer'