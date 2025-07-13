"""
File: smartcash/ui/core/ui_module.py
Description: Central UIModule hub for all module functionality with shared method registry.
"""

from typing import Dict, Any, Optional, Callable, List, Union, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.handlers.ui_handler import UIHandler, ModuleUIHandler
from smartcash.ui.core.handlers.config_handler import SharedConfigHandler
from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult, OperationStatus
from smartcash.ui.core.errors.exceptions import SmartCashUIError
from smartcash.ui.core.errors.context import ErrorContext
from smartcash.ui.core.decorators.error_decorators import handle_errors

T = TypeVar('T')

class ModuleStatus(Enum):
    """Status for UIModule."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    CLEANUP = "cleanup"

@dataclass
class ModuleInfo:
    """Information about a UIModule."""
    name: str
    parent_module: Optional[str]
    status: ModuleStatus
    created_at: datetime
    initialized_at: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    components: Dict[str, str] = field(default_factory=dict)
    operations: List[str] = field(default_factory=list)


class SharedMethodRegistry:
    """Central registry for shared methods across UIModules."""
    
    _shared_methods: Dict[str, Callable] = {}
    _method_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_method(cls, 
                       name: str, 
                       method: Callable, 
                       overwrite: bool = False,
                       description: str = None,
                       category: str = "general") -> None:
        """Register a method for sharing across modules.
        
        Args:
            name: Method name
            method: Callable method
            overwrite: Allow overwriting existing method
            description: Method description
            category: Method category for organization
            
        Raises:
            ValueError: If method already exists and overwrite=False
        """
        if name in cls._shared_methods and not overwrite:
            raise ValueError(f"Method '{name}' already registered. Use overwrite=True to replace.")
        
        cls._shared_methods[name] = method
        cls._method_metadata[name] = {
            'description': description or f"Shared method: {name}",
            'category': category,
            'registered_at': datetime.now().isoformat(),
            'callable_name': getattr(method, '__name__', str(method))
        }
        
        logger = get_module_logger("smartcash.ui.core.shared_registry")
        logger.debug(f"🔗 Registered shared method: {name} ({category})")
    
    @classmethod  
    def get_method(cls, name: str) -> Optional[Callable]:
        """Get a shared method by name.
        
        Args:
            name: Method name
            
        Returns:
            Callable method or None if not found
        """
        return cls._shared_methods.get(name)
    
    @classmethod
    def list_methods(cls, category: str = None) -> Dict[str, Dict[str, Any]]:
        """List all registered methods.
        
        Args:
            category: Filter by category (optional)
            
        Returns:
            Dictionary of method_name -> metadata
        """
        if category is None:
            return cls._method_metadata.copy()
        
        return {
            name: metadata 
            for name, metadata in cls._method_metadata.items()
            if metadata.get('category') == category
        }
    
    @classmethod
    def inject_methods(cls, target_module: 'UIModule', category: str = None) -> int:
        """Inject shared methods into a UIModule.
        
        Args:
            target_module: UIModule instance to inject methods into
            category: Only inject methods from this category (optional)
            
        Returns:
            Number of methods injected
        """
        injected = 0
        
        for name, method in cls._shared_methods.items():
            metadata = cls._method_metadata.get(name, {})
            
            # Filter by category if specified
            if category and metadata.get('category') != category:
                continue
                
            try:
                target_module.share_method(name, method)
                injected += 1
            except Exception as e:
                logger = get_module_logger("smartcash.ui.core.shared_registry")
                logger.warning(f"Failed to inject method '{name}': {e}")
        
        logger = get_module_logger("smartcash.ui.core.shared_registry")
        logger.debug(f"🔗 Injected {injected} shared methods into {target_module.module_name}")
        return injected
    
    @classmethod
    def unregister_method(cls, name: str) -> bool:
        """Unregister a shared method.
        
        Args:
            name: Method name to unregister
            
        Returns:
            True if method was unregistered, False if not found
        """
        if name in cls._shared_methods:
            del cls._shared_methods[name]
            del cls._method_metadata[name]
            
            logger = get_module_logger("smartcash.ui.core.shared_registry")
            logger.debug(f"🗑️ Unregistered shared method: {name}")
            return True
        
        return False
    
    @classmethod
    def clear_category(cls, category: str) -> int:
        """Clear all methods in a category.
        
        Args:
            category: Category to clear
            
        Returns:
            Number of methods removed
        """
        methods_to_remove = [
            name for name, metadata in cls._method_metadata.items()
            if metadata.get('category') == category
        ]
        
        for name in methods_to_remove:
            cls.unregister_method(name)
        
        logger = get_module_logger("smartcash.ui.core.shared_registry")
        logger.debug(f"🗑️ Cleared {len(methods_to_remove)} methods from category: {category}")
        return len(methods_to_remove)


class UIModule:
    """Central hub for all module functionality.
    
    Consolidates:
    - UI component management via UIHandler composition
    - Configuration handling via ConfigHandler composition  
    - Operation execution via OperationHandler composition
    - State management and lifecycle
    - Event coordination and method sharing
    
    This class acts as a facade over the existing handler architecture,
    providing a unified interface while preserving all existing functionality.
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: str = None,
                 config: Dict[str, Any] = None,
                 auto_initialize: bool = False):
        """Initialize UIModule.
        
        Args:
            module_name: Module name (e.g., 'downloader')
            parent_module: Parent module (e.g., 'dataset')
            config: Initial configuration
            auto_initialize: Auto-initialize handlers
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Setup logger
        self.logger = get_module_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Core handlers (composition over inheritance)
        self._ui_handler: Optional[ModuleUIHandler] = None
        self._config_handler: Optional[SharedConfigHandler] = None
        self._operation_handler: Optional[OperationHandler] = None
        
        # Module state
        self._status = ModuleStatus.PENDING
        self._created_at = datetime.now()
        self._initialized_at: Optional[datetime] = None
        self._error_count = 0
        self._last_error: Optional[str] = None
        
        # Component and operation registry
        self._components: Dict[str, Any] = {}
        self._operations: Dict[str, Callable] = {}
        self._shared_methods: Dict[str, Callable] = {}
        
        # Initialize with config
        self._config = config or {}
        
        # Auto-initialize if requested
        if auto_initialize:
            self.initialize()
        
        self.logger.debug(f"🚀 Created UIModule: {self.full_module_name}")
    
    # === Initialization and Lifecycle ===
    
    @handle_errors(context_attr='_create_error_context')
    def initialize(self, config: Dict[str, Any] = None) -> 'UIModule':
        """Initialize UIModule with handlers and components.
        
        Args:
            config: Configuration dictionary (optional)
            
        Returns:
            Self for method chaining
            
        Raises:
            SmartCashUIError: If initialization fails
        """
        if self._status != ModuleStatus.PENDING:
            self.logger.warning(f"UIModule {self.full_module_name} already initialized")
            return self
        
        self._status = ModuleStatus.INITIALIZING
        
        try:
            # Merge configuration
            if config:
                self._config.update(config)
            
            # Initialize core handlers
            self._initialize_handlers()
            
            # Setup initial components
            self._setup_components()
            
            # Register default operations
            self._register_default_operations()
            
            # Inject shared methods
            SharedMethodRegistry.inject_methods(self)
            
            # Mark as ready
            self._status = ModuleStatus.READY
            self._initialized_at = datetime.now()
            
            self.logger.info(f"✅ Initialized UIModule: {self.full_module_name}")
            
        except Exception as e:
            self._status = ModuleStatus.ERROR
            self._last_error = str(e)
            self._error_count += 1
            
            error_msg = f"Failed to initialize UIModule {self.full_module_name}: {e}"
            self.logger.error(error_msg)
            raise SmartCashUIError(error_msg, context=self._create_error_context())
        
        return self
    
    def _initialize_handlers(self) -> None:
        """Initialize core handlers with composition."""
        # UI Handler for component management
        self._ui_handler = ModuleUIHandler(
            module_name=self.module_name,
            parent_module=self.parent_module
        )
        
        # Config Handler for configuration management
        self._config_handler = SharedConfigHandler(
            module_name=self.module_name,
            parent_module=self.parent_module
        )
        
        # Operation Handler for operation execution
        # Note: We'll create this as a basic implementation
        # since the actual OperationHandler is abstract
        self._operation_handler = self._create_operation_handler()
        
        self.logger.debug(f"🔧 Initialized handlers for {self.full_module_name}")
    
    def _create_operation_handler(self) -> OperationHandler:
        """Create concrete OperationHandler implementation."""
        
        class ConcreteOperationHandler(OperationHandler):
            def __init__(self, module_name: str, parent_module: str = None):
                super().__init__(module_name, parent_module)
                self._operations = {}
            
            def get_operations(self) -> Dict[str, Callable]:
                return self._operations.copy()
            
            def register_operation(self, name: str, operation: Callable) -> None:
                self._operations[name] = operation
            
            def clear_outputs(self) -> None:
                """Clear operation outputs (required by base class)."""
                # Implementation for clearing outputs
                pass
            
            def update_status(self, message: str, status_type: str = 'info') -> None:
                """Update status (required by base class)."""
                # Implementation for status updates
                pass
        
        return ConcreteOperationHandler(
            module_name=self.module_name,
            parent_module=self.parent_module
        )
    
    def _setup_components(self) -> None:
        """Setup initial UI components."""
        # Components will be registered as they are created
        # This method can be overridden by subclasses
        pass
    
    def _register_default_operations(self) -> None:
        """Register default operations."""
        # Register basic operations
        self.register_operation('initialize', lambda: self.get_info())
        self.register_operation('get_status', lambda: self.get_status())
        self.register_operation('reset', lambda: self.reset())
    
    def _create_error_context(self) -> ErrorContext:
        """Create error context for this module."""
        return ErrorContext(
            component=f"UIModule.{self.full_module_name}",
            operation="",
            details={
                'module_name': self.module_name,
                'parent_module': self.parent_module,
                'status': self._status.value,
                'error_count': self._error_count
            }
        )
    
    # === Component Management ===
    
    def register_component(self, component_type: str, component: Any) -> None:
        """Register a UI component.
        
        Args:
            component_type: Type/name of component
            component: Component instance
        """
        self._components[component_type] = component
        
        # Also register with UI handler for container-aware access
        if self._ui_handler:
            self._ui_handler.ui_components[component_type] = component
        
        self.logger.debug(f"📦 Registered component: {component_type} in {self.full_module_name}")
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """Get a UI component.
        
        Args:
            component_type: Type/name of component
            
        Returns:
            Component instance or None
        """
        # Try direct lookup first
        component = self._components.get(component_type)
        
        # Fall back to UI handler's container-aware search
        if component is None and self._ui_handler:
            component = self._ui_handler._find_component(component_type)
        
        return component
    
    def list_components(self) -> Dict[str, str]:
        """List all registered components.
        
        Returns:
            Dictionary of component_type -> component_class_name
        """
        return {
            component_type: component.__class__.__name__
            for component_type, component in self._components.items()
        }
    
    # === Operation Management ===
    
    def register_operation(self, name: str, operation: Callable) -> None:
        """Register an operation.
        
        Args:
            name: Operation name
            operation: Callable operation
        """
        self._operations[name] = operation
        
        # Also register with operation handler
        if self._operation_handler and hasattr(self._operation_handler, 'register_operation'):
            self._operation_handler.register_operation(name, operation)
        
        self.logger.debug(f"⚙️ Registered operation: {name} in {self.full_module_name}")
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> OperationResult:
        """Execute an operation by name.
        
        Args:
            operation_name: Name of operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            OperationResult with execution details
        """
        if operation_name not in self._operations:
            return OperationResult(
                status=OperationStatus.FAILED,
                error=ValueError(f"Unknown operation: {operation_name}"),
                message=f"Operation '{operation_name}' not found in {self.full_module_name}"
            )
        
        if self._operation_handler:
            return self._operation_handler.execute_named_operation(operation_name, *args, **kwargs)
        else:
            # Fallback execution
            try:
                result = self._operations[operation_name](*args, **kwargs)
                return OperationResult(
                    status=OperationStatus.COMPLETED,
                    data=result,
                    message=f"Operation '{operation_name}' completed successfully"
                )
            except Exception as e:
                return OperationResult(
                    status=OperationStatus.FAILED,
                    error=e,
                    message=f"Operation '{operation_name}' failed: {str(e)}"
                )
    
    def list_operations(self) -> List[str]:
        """List all registered operations.
        
        Returns:
            List of operation names
        """
        return list(self._operations.keys())
    
    # === Configuration Management ===
    
    def update_config(self, **config_updates) -> None:
        """Update module configuration.
        
        Args:
            **config_updates: Configuration key-value pairs
        """
        self._config.update(config_updates)
        
        # Also update config handler
        if self._config_handler:
            try:
                self._config_handler.update_config(config_updates)
            except Exception as e:
                self.logger.warning(f"Failed to update config handler: {e}")
        
        self.logger.debug(f"🔧 Updated config for {self.full_module_name}: {list(config_updates.keys())}")
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (None for all config)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self._config.copy()
        
        return self._config.get(key, default)
    
    def reset_config(self) -> None:
        """Reset configuration to empty state."""
        self._config.clear()
        
        if self._config_handler:
            try:
                self._config_handler.reset_config()
            except Exception as e:
                self.logger.warning(f"Failed to reset config handler: {e}")
        
        self.logger.debug(f"🔄 Reset config for {self.full_module_name}")
    
    # === Shared Method Management ===
    
    def share_method(self, method_name: str, method: Callable) -> None:
        """Share a method within this module.
        
        Args:
            method_name: Name to assign to the method
            method: Callable method
        """
        self._shared_methods[method_name] = method
        setattr(self, method_name, method)
        
        self.logger.debug(f"🔗 Shared method: {method_name} in {self.full_module_name}")
    
    def get_shared_method(self, method_name: str) -> Optional[Callable]:
        """Get a shared method.
        
        Args:
            method_name: Method name
            
        Returns:
            Callable method or None
        """
        return self._shared_methods.get(method_name)
    
    def list_shared_methods(self) -> List[str]:
        """List all shared methods.
        
        Returns:
            List of shared method names
        """
        return list(self._shared_methods.keys())
    
    # === UI Delegation Methods ===
    
    def update_status(self, message: str, status_type: str = 'info') -> None:
        """Update status display."""
        if self._ui_handler:
            self._ui_handler.update_status(message, status_type)
        else:
            getattr(self.logger, status_type, self.logger.info)(f"[Status] {message}")
    
    def update_progress(self, value: float, message: str = None) -> None:
        """Update progress display."""
        if self._ui_handler:
            self._ui_handler.update_progress(value, message)
        else:
            self.logger.info(f"📊 Progress: {value*100:.1f}% - {message or 'Processing...'}")
    
    def log_message(self, message: str, level: str = 'info') -> None:
        """Log a message."""
        if self._ui_handler:
            self._ui_handler.log_message(message, level)
        else:
            getattr(self.logger, level, self.logger.info)(message)
    
    def show_dialog(self, title: str, message: str, dialog_type: str = 'info') -> None:
        """Show a dialog."""
        if self._ui_handler:
            self._ui_handler.show_dialog(title, message, dialog_type)
        else:
            self.logger.info(f"💬 Dialog: {title} - {message}")
    
    def clear_components(self) -> None:
        """Clear all UI components."""
        if self._ui_handler:
            self._ui_handler.clear_all_components()
        
        self._components.clear()
        self.logger.debug(f"🧹 Cleared components for {self.full_module_name}")
    
    # === State and Information ===
    
    def get_status(self) -> ModuleStatus:
        """Get current module status.
        
        Returns:
            Current ModuleStatus
        """
        return self._status
    
    def get_info(self) -> ModuleInfo:
        """Get comprehensive module information.
        
        Returns:
            ModuleInfo with current state
        """
        return ModuleInfo(
            name=self.module_name,
            parent_module=self.parent_module,
            status=self._status,
            created_at=self._created_at,
            initialized_at=self._initialized_at,
            error_count=self._error_count,
            last_error=self._last_error,
            components=self.list_components(),
            operations=self.list_operations()
        )
    
    def is_ready(self) -> bool:
        """Check if module is ready for use.
        
        Returns:
            True if module is ready
        """
        return self._status == ModuleStatus.READY
    
    def has_errors(self) -> bool:
        """Check if module has errors.
        
        Returns:
            True if module has errors
        """
        return self._error_count > 0
    
    # === Utility Methods ===
    
    def reset(self) -> None:
        """Reset module to initial state."""
        # Clear components and operations
        self.clear_components()
        self._operations.clear()
        self._shared_methods.clear()
        
        # Reset configuration
        self.reset_config()
        
        # Reset error state
        self._error_count = 0
        self._last_error = None
        
        # Cleanup handlers
        if self._ui_handler:
            self._ui_handler.cleanup()
        if self._config_handler:
            self._config_handler.cleanup()
        if self._operation_handler:
            self._operation_handler.cleanup()
        
        # Reset status
        self._status = ModuleStatus.PENDING
        self._initialized_at = None
        
        self.logger.info(f"🔄 Reset UIModule: {self.full_module_name}")
    
    def cleanup(self) -> None:
        """Cleanup module resources."""
        self._status = ModuleStatus.CLEANUP
        
        try:
            # Cleanup handlers
            if self._ui_handler:
                self._ui_handler.cleanup()
            if self._config_handler:
                self._config_handler.cleanup()
            if self._operation_handler:
                self._operation_handler.cleanup()
            
            # Clear all data
            self._components.clear()
            self._operations.clear()
            self._shared_methods.clear()
            self._config.clear()
            
            self.logger.debug(f"🧹 Cleaned up UIModule: {self.full_module_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")
    
    # === Context Manager Support ===
    
    def __enter__(self) -> 'UIModule':
        """Enter context manager."""
        if not self.is_ready():
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.cleanup()
    
    # === String Representation ===
    
    def __repr__(self) -> str:
        return f"UIModule(name='{self.full_module_name}', status='{self._status.value}', components={len(self._components)}, operations={len(self._operations)})"
    
    def __str__(self) -> str:
        return f"UIModule[{self.full_module_name}]: {self._status.value}"


# === Convenience Functions ===

def register_ui_method(name: str, 
                      method: Callable, 
                      category: str = "ui",
                      description: str = None) -> None:
    """Register a UI-related shared method.
    
    Args:
        name: Method name
        method: Callable method
        category: Method category (default: 'ui')
        description: Method description
    """
    SharedMethodRegistry.register_method(
        name=name,
        method=method,
        category=category,
        description=description or f"UI method: {name}"
    )

def register_operation_method(name: str, 
                            method: Callable, 
                            description: str = None) -> None:
    """Register an operation-related shared method.
    
    Args:
        name: Method name
        method: Callable method
        description: Method description
    """
    SharedMethodRegistry.register_method(
        name=name,
        method=method,
        category="operations",
        description=description or f"Operation method: {name}"
    )

def register_config_method(name: str, 
                         method: Callable, 
                         description: str = None) -> None:
    """Register a configuration-related shared method.
    
    Args:
        name: Method name
        method: Callable method
        description: Method description
    """
    SharedMethodRegistry.register_method(
        name=name,
        method=method,
        category="config",
        description=description or f"Config method: {name}"
    )