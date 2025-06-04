"""
File: smartcash/ui/setup/dependency/__init__.py
Deskripsi: Public API untuk dependency installer module
"""

# Import main initializer
from smartcash.ui.setup.dependency.dependency_init import (
    initialize_dependency_ui,
    DependencyInitializer,
    validate_dependency_setup,
    get_dependency_config_handler,
    get_dependency_status,
)

# Export public API
__all__ = [
    'initialize_dependency_ui',
    'DependencyInitializer',
    'validate_dependency_setup',
    'get_dependency_config_handler',
    'get_dependency_status',
]

# Convenience aliases
initialize_ui = initialize_dependency_ui
validate_setup = validate_dependency_setup
get_config_handler = get_dependency_config_handler
get_status = get_dependency_status
