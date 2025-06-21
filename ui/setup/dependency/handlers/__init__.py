"""
File: smartcash/ui/setup/dependency/handlers/__init__.py
Deskripsi: Dependency handlers module exports tanpa check/uncheck functionality
"""

from .config_extractor import extract_dependency_config, validate_extracted_config, get_config_summary
from .config_updater import update_dependency_ui, reset_dependency_ui, apply_config_to_ui
from .defaults import get_default_dependency_config, get_minimal_config, get_environment_specific_config
from .dependency_handler import (
    setup_dependency_handlers,
    extract_current_config,
    apply_config_to_ui as apply_config,
    reset_ui_to_defaults,
    validate_ui_components,
    get_handlers_status
)
from .installation_handler import setup_installation_handler
from .analysis_handler import setup_analysis_handler
from .status_check_handler import setup_status_check_handler

__all__ = [
    # Config handlers
    'extract_dependency_config',
    'update_dependency_ui',
    'reset_dependency_ui',
    'get_default_dependency_config',
    'get_minimal_config',
    'get_environment_specific_config',
    
    # Main handler setup
    'setup_dependency_handlers',
    'extract_current_config',
    'apply_config_to_ui',
    'apply_config',
    'reset_ui_to_defaults',
    
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