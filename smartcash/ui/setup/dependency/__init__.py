"""
Dependency management module for SmartCash UI setup.

This module provides the public API for managing Python package dependencies,
including installation, verification, and status reporting.
"""

# Core functionality
from .dependency_initializer import initialize_dependency_ui, get_dependency_config

# Package management utilities
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

# Configuration handling
from .handlers.config_handler import DependencyConfigHandler
from .handlers.defaults import get_default_dependency_config

# Public API
__all__ = [
    # Core functionality
    'initialize_dependency_ui',
    'get_dependency_config',
    
    # Configuration
    'DependencyConfigHandler',
    'get_default_dependency_config',
    
    # Package management
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

# Package metadata
__version__ = '1.0.0'
__author__ = 'SmartCash Team'