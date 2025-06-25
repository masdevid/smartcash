"""
File: smartcash/ui/setup/dependency/handlers/__init__.py
Deskripsi: Dependency handlers module exports

This module provides a clean interface to all dependency management functionality,
including configuration handling, UI updates, and package management.
"""

from smartcash.ui.setup.dependency.handlers.installation_handler import setup_installation_handler

from smartcash.ui.setup.dependency.handlers.config_extractor import (
    extract_dependency_config,
    validate_extracted_config,
    get_config_summary
)
from smartcash.ui.setup.dependency.handlers.config_updater import (
    update_dependency_ui,
    reset_dependency_ui,
    apply_config_to_ui
)
from smartcash.ui.setup.dependency.handlers.defaults import (
    DEFAULT_CONFIG,
    PACKAGE_CATEGORIES,
)
from smartcash.ui.setup.dependency.handlers.dependency_handler import (
    setup_dependency_handlers,
    extract_current_config,
    apply_config_to_ui,
    reset_ui_to_defaults,
    validate_ui_components,
    get_handlers_status
)
from smartcash.ui.setup.dependency.handlers.installation_handler import setup_installation_handler
from smartcash.ui.setup.dependency.handlers.analysis_handler import setup_analysis_handler
from smartcash.ui.setup.dependency.handlers.status_check_handler import setup_status_check_handler

__all__ = [
    # Configuration
    'DEFAULT_CONFIG',
    'PACKAGE_CATEGORIES',
    'extract_dependency_config',
    'update_dependency_ui',
    'reset_dependency_ui',
    'get_minimal_config',
    
    # Main handler setup
    'setup_dependency_handlers',
    'extract_current_config',
    'apply_config_to_ui',
    'reset_ui_to_defaults',
    'validate_ui_components',
    'get_handlers_status',
    
    # Individual handlers
    'setup_installation_handler',
    'setup_analysis_handler', 
    'setup_status_check_handler',
    
    # Validation
    'validate_extracted_config',
    'validate_ui_components',
    'get_config_summary',
    'get_handlers_status'
]