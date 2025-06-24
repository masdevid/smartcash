"""
File: smartcash/ui/setup/dependency/__init__.py
Deskripsi: Main dependency module exports dengan public API yang lengkap
"""

from .dependency_initializer import (
    initialize_dependency_ui,
    get_dependency_config,
    get_dependency_status,
    cleanup_dependency_resources
)

# Import utility functions untuk backward compatibility
from .utils import (
    get_installed_packages_dict,
    check_package_installation_status,
    install_single_package,
    batch_check_packages_status,
    create_operation_context,
    update_status_panel,
    log_to_ui_safe,
    generate_comprehensive_status_report
)

# Config dan status utilities
from .handlers.config_handler import DependencyConfigHandler
from .handlers.defaults import get_default_dependency_config

# Main public API
__all__ = [
    # Core initialization
    'initialize_dependency_ui',
    'get_dependency_config',
    'get_dependency_status',
    'cleanup_dependency_resources',
    
    # Config handling
    'DependencyConfigHandler',
    'get_default_dependency_config',
    
    # Package utilities
    'get_installed_packages_dict',
    'check_package_installation_status',
    'install_single_package',
    'batch_check_packages_status',
    
    # UI utilities
    'create_operation_context',
    'update_status_panel',
    'log_to_ui_safe',
    
    # Reporting
    'generate_comprehensive_status_report'
]

# Version info
__version__ = '1.0.0'
__author__ = 'SmartCash Team'