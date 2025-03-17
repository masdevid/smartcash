"""
File: smartcash/ui/handlers/__init__.py
Deskripsi: Import handler yang umum digunakan untuk mengelola komponen UI
"""

from smartcash.ui.handlers.environment_handler import (
    detect_environment, filter_drive_tree, fallback_get_directory_tree, 
    check_smartcash_dir, sync_configs
)
from smartcash.ui.handlers.error_handler import (
    setup_error_handlers, handle_error, create_error_handler, try_except_decorator
)
from smartcash.ui.handlers.observer_handler import (
    setup_observer_handlers, register_ui_observer, create_progress_observer
)
from smartcash.ui.handlers.config_handler import (
    setup_config_handlers, handle_config_load, handle_config_save, 
    update_config, get_config_value, set_config_value
)

__all__ = [
    # Environment handlers
    'detect_environment', 'filter_drive_tree', 'fallback_get_directory_tree',
    'check_smartcash_dir', 'sync_configs',
    
    # Error handlers
    'setup_error_handlers', 'handle_error', 'create_error_handler', 'try_except_decorator',
    
    # Observer handlers
    'setup_observer_handlers', 'register_ui_observer', 'create_progress_observer',
    
    # Config handlers
    'setup_config_handlers', 'handle_config_load', 'handle_config_save',
    'update_config', 'get_config_value', 'set_config_value'
]