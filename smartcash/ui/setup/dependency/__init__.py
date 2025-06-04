"""
File: smartcash/ui/setup/dependency/__init__.py
Deskripsi: Main dependency module exports dengan public API
"""

from smartcash.ui.setup.dependency.dependency_init import (
    initialize_dependency_ui,
    get_dependency_config,
    get_dependency_config_handler,
    update_dependency_config,
    reset_dependency_config,
    validate_dependency_setup,
    get_dependency_status,
    cleanup_dependency_generators,
    get_selected_packages_count,
    get_installation_settings,
    get_analysis_settings,
    is_auto_analyze_enabled,
    debug_generator_count,
    debug_button_status,
    debug_config_summary
)

# Main public API
__all__ = [
    # Core functions
    'initialize_dependency_ui',
    'get_dependency_config',
    'get_dependency_config_handler',
    
    # Config management
    'update_dependency_config',
    'reset_dependency_config',
    
    # Status and validation
    'validate_dependency_setup',
    'get_dependency_status',
    'cleanup_dependency_generators',
    
    # Config utilities
    'get_selected_packages_count',
    'get_installation_settings',
    'get_analysis_settings',
    'is_auto_analyze_enabled',
    
    # Debug utilities
    'debug_generator_count',
    'debug_button_status',
    'debug_config_summary'
]
