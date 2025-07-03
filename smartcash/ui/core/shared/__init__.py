"""
File: smartcash/ui/core/shared/__init__.py
Deskripsi: Export shared utilities untuk SmartCash UI core module
"""

# Import shared components
from smartcash.ui.core.shared.logger import EnhancedUILogger, get_enhanced_logger
from smartcash.ui.core.shared.error_handler import (
    CoreErrorHandler,
    ErrorLevel,
    ErrorContext,
    get_error_handler,
    handle_component_validation,
    safe_component_operation,
    validate_ui_components,
    handle_errors
)
from smartcash.ui.core.shared.ui_component_manager import (
    UIComponentManager,
    ComponentRegistry,
    get_component_manager
)
from smartcash.ui.core.shared.core_shared_config_manager import (
    CoreSharedConfigManager,
    get_core_shared_manager
)

# Public exports
__all__ = [
    # Logger
    'EnhancedUILogger',
    'get_enhanced_logger',
    
    # Error handler
    'CoreErrorHandler',
    'ErrorLevel',
    'ErrorContext',
    'get_error_handler',
    'handle_component_validation',
    'safe_component_operation',
    'validate_ui_components',
    'handle_errors',
    
    # Component manager
    'UIComponentManager',
    'ComponentRegistry',
    'get_component_manager',
    'ComponentRegistry',
    'get_component_manager',
    
    # Config manager
    'CoreSharedConfigManager',
    'get_core_shared_manager',
]