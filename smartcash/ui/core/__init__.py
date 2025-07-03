"""
File: smartcash/ui/core/__init__.py
Deskripsi: Core UI module dengan fail-fast principle
"""

# Handlers
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.handlers.ui_handler import UIHandler

# Initializers
from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.initializers.operation_initializer import OperationInitializer

# Shared utilities
from smartcash.ui.core.shared.ui_component_manager import UIComponentManager, ComponentRegistry

__all__ = [
    'BaseHandler',
    'UIHandler', 
    'BaseInitializer',
    'OperationInitializer',
    'UIComponentManager',
    'ComponentRegistry'
]