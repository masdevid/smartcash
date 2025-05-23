"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Initializer untuk instalasi dependencies dengan alur otomatis yang lebih robust
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.dependency_installer.components import create_dependency_installer_ui
from smartcash.ui.setup.dependency_installer.handlers import setup_dependency_installer_handlers
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import create_ui_logger

# Konstanta untuk namespace logger
DEPENDENCY_INSTALLER_LOGGER_NAMESPACE = "smartcash.setup.dependency_installer"
# Konstanta untuk ID namespace di UI
MODULE_LOGGER_NAME = "DEP-INSTALLER"

def initialize_dependency_installer() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk instalasi dependencies.
    
    Returns:
        Dictionary UI components
    """
    # Inisialisasi environment manager
    env_manager = get_environment_manager()
    
    # Inisialisasi config manager dengan base_dir dari environment manager
    config_manager = get_config_manager(base_dir=env_manager.base_dir)
    
    # Buat komponen UI
    ui_components = create_dependency_installer_ui(env_manager, config_manager)
    
    # Setup logger dengan namespace spesifik
    logger = create_ui_logger(ui_components, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    ui_components['logger'] = logger
    ui_components['logger_namespace'] = DEPENDENCY_INSTALLER_LOGGER_NAMESPACE
    ui_components['dependency_installer_initialized'] = True
    
    # Setup handlers
    setup_dependency_installer_handlers(ui_components, env_manager, config_manager)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components
