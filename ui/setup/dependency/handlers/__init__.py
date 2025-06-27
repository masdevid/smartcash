# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/__init__.py - UPDATED
# Deskripsi: Complete handlers module exports untuk dependency management
# =============================================================================

# Main coordinator
from .event_handlers import setup_all_handlers

# Base class
from .base_handler import BaseDependencyHandler

# SRP handlers dengan base class pattern
from .config_event_handlers import setup_config_handlers, ConfigEventHandler
from .operation_handlers import setup_operation_handlers, OperationHandler
from .selection_handlers import setup_selection_handlers, SelectionHandler

# Config management
from .config_handler import DependencyConfigHandler
from .config_extractor import extract_dependency_config
from .config_updater import update_dependency_ui, reset_dependency_ui
from .defaults import get_default_dependency_config, get_all_packages, get_package_by_key

__all__ = [
    # Main coordinator
    'setup_all_handlers',
    
    # Base class
    'BaseDependencyHandler',
    
    # SRP handlers dengan base class
    'setup_config_handlers', 'ConfigEventHandler',
    'setup_operation_handlers', 'OperationHandler',
    'setup_selection_handlers', 'SelectionHandler',
    
    # Config management
    'DependencyConfigHandler',
    'extract_dependency_config',
    'update_dependency_ui',
    'reset_dependency_ui',
    'get_default_dependency_config',
    'get_all_packages',
    'get_package_by_key'
]