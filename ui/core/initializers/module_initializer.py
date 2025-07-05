"""
File: smartcash/ui/core/initializers/module_initializer.py
Deskripsi: Module-specific initializer dengan UI handler integration dan full lifecycle support.
Kombinasi ConfigurableInitializer + ModuleUIHandler untuk complete module initialization.
"""

import contextlib
import time
from typing import Dict, Any, Optional, Type, List, Callable, Tuple
from pathlib import Path

from smartcash.ui.core.initializers.config_initializer import ConfigurableInitializer
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler as LegacyConfigHandler


class ModuleInitializer(ConfigurableInitializer):
    """Module-specific initializer dengan complete UI dan config management.
    
    Features:
    - 🎯 Complete module initialization
    - 🔧 Auto handler setup
    - 📊 UI-Config synchronization
    - 🔄 Lifecycle management
    - 💾 Persistence support
    - 🌐 Module instance management
    """
    
    # Class-level storage for module instances
    _module_instances = {}
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 handler_class: Optional[Type] = None,
                 config_handler_class: Optional[Type] = None,
                 enable_shared_config: bool = True,
                 auto_setup_handlers: bool = True):
        """Initialize module initializer.
        
        Args:
            module_name: Nama module
            parent_module: Parent module untuk organization
            handler_class: Custom UI handler class (default: ModuleUIHandler)
            config_handler_class: Custom config handler untuk backward compat
            enable_shared_config: Enable config sharing
            auto_setup_handlers: Auto setup UI event handlers
        """
        # Initialize parent
        super().__init__(module_name, parent_module, config_handler_class, enable_shared_config)
        
        # Handler setup
        self._handler_class = handler_class or ModuleUIHandler
        self._auto_setup_handlers = auto_setup_handlers
        self._module_handler: Optional[ModuleUIHandler] = None
        
        # Additional handlers registry
        self._operation_handlers: Dict[str, Any] = {}
        self._service_handlers: Dict[str, Any] = {}
        
        # Register instance
        self._instance_id = f"{parent_module}.{module_name}" if parent_module else module_name
        self.__class__._module_instances[self._instance_id] = self
        
        self.logger.debug(f"🎯 ModuleInitializer created with handler: {self._handler_class.__name__}")
    
    # === Handler Creation ===
    
    def create_module_handler(self) -> ModuleUIHandler:
        """Create module handler instance. Override untuk custom handler."""
        return self._handler_class(
            module_name=self.module_name,
            parent_module=self.parent_module,
            default_config=self.get_default_config(),
            auto_setup_handlers=self._auto_setup_handlers,
            enable_sharing=True
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration. Override di subclass."""
        return {}
    
    # === Module Instance Management ===
    
    @classmethod
    def get_module_instance(cls, module_name: str, parent_module: Optional[str] = None, **kwargs) -> 'ModuleInitializer':
        """Get or create a module instance.
        
        Args:
            module_name: Name of the module
            parent_module: Optional parent module name
            **kwargs: Additional arguments passed to the constructor
            
        Returns:
            ModuleInitializer: The module instance
        """
        instance_id = f"{parent_module}.{module_name}" if parent_module else module_name
        
        if instance_id not in cls._module_instances:
            cls._module_instances[instance_id] = cls(
                module_name=module_name,
                parent_module=parent_module,
                **kwargs
            )
            
        return cls._module_instances[instance_id]
    
    @classmethod
    def initialize_module_ui(cls, 
                           module_name: str,
                           parent_module: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None,
                           initializer_class: Optional[Type['ModuleInitializer']] = None,
                           **kwargs) -> Any:
        """Initialize a module UI with proper error handling.
        
        Args:
            module_name: Name of the module to initialize
            parent_module: Optional parent module name
            config: Optional configuration dictionary
            initializer_class: Custom initializer class (default: cls)
            **kwargs: Additional arguments passed to the initializer
            
        Returns:
            Any: The UI component or error UI
        """
        try:
            # Get or create module instance
            initializer = (initializer_class or cls).get_module_instance(
                module_name=module_name,
                parent_module=parent_module,
                **kwargs
            )
            
            # Initialize with config
            result = initializer.initialize(config=config)
            
            # Return UI component
            if result.get('success', False):
                return result.get('ui_components', {}).get('ui')
            else:
                error_msg = result.get('error', 'Unknown initialization error')
                raise RuntimeError(error_msg)
                
        except Exception as e:
            # Handle initialization error
            try:
                # Try to create an instance of the provided initializer class or use the default
                initializer_cls = initializer_class or cls
                error_handler = initializer_cls(
                    module_name=module_name,
                    parent_module=parent_module,
                    **kwargs
                )
                error_result = error_handler.handle_initialization_error(e, f"initializing {module_name} UI")
                return error_result.get('ui')
            except Exception as inner_e:
                # If we can't create the error handler, return a simple error message
                import traceback
                error_msg = f"Failed to initialize {module_name} UI and error handling failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                from IPython.display import HTML
                return HTML(f'<div style="color: red; padding: 10px; border: 1px solid red;">{error_msg}</div>')
    
    # === Error Handling ===
    
    def handle_initialization_error(self, error: Exception, context: str = "initializing module") -> Dict[str, Any]:
        """Handle initialization errors and return error UI."""
        from smartcash.ui.core.shared.error_handler import CoreErrorHandler
        
        error_msg = str(error)
        self.logger.error(f"❌ Error {context}: {error_msg}", exc_info=True)
        
        # Create error UI
        error_handler = CoreErrorHandler(self.module_name)
        error_ui = error_handler.create_error_ui({
            'error': True,
            'message': f"Error {context}: {error_msg}",
            'title': f"{self.module_name.capitalize()} Initialization Error"
        })
        
        # Return error result
        return {
            'success': False,
            'error': error_msg,
            'ui': error_ui['ui'] if isinstance(error_ui, dict) and 'ui' in error_ui else error_ui
        }
    
    # === Extended Setup ===
    
    def setup_handlers(self, **kwargs) -> None:
        """Setup module handler dan additional handlers."""
        self.logger.info("🔧 Setting up module handlers...")
        
        try:
            # 1. Create main module handler
            self._module_handler = self.create_module_handler()
            
            # 2. Setup dengan UI components
            self._module_handler.setup(self._ui_components)
            
            # 3. Register di handlers dict
            self._handlers['module'] = self._module_handler
            self._handlers['config'] = self._module_handler  # Alias untuk backward compat
            
            # 4. Setup operation handlers
            self.setup_operation_handlers()
            
            # 5. Setup service handlers  
            self.setup_service_handlers()
            
            # 6. Custom handler setup
            self.setup_custom_handlers()
            
            self.logger.info(f"✅ Handlers setup complete: {len(self._handlers)} handlers")
            
        except Exception as e:
            raise RuntimeError(f"Handler setup failed: {str(e)}") from e
    
    def setup_operation_handlers(self) -> None:
        """Setup operation-specific handlers. Override di subclass."""
        pass
    
    def setup_service_handlers(self) -> None:
        """Setup service/business logic handlers. Override di subclass."""
        pass
    
    @contextlib.contextmanager
    def _log_step(self, step_name: str):
        """Context manager untuk menangani logging step dengan konsisten
        
        Args:
            step_name: Nama step yang akan di-log
            
        Yields:
            None
            
        Example:
            with self._log_step("Processing data"):
                # kode yang ingin di-log waktunya
                pass
        """
        self.logger.info(f"🔍 {step_name}...")
        start_time = time.time()
        try:
            yield
            elapsed = time.time() - start_time
            self.logger.info(f"✅ {step_name} selesai ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"❌ Gagal {step_name.lower()} setelah {elapsed:.2f}s: {str(e)}")
            raise
    
    def setup_custom_handlers(self) -> None:
        """Override ini untuk menambahkan custom handlers."""
        pass
    
    # === UI Component Creation Helper ===
    
    def create_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Default implementation yang bisa di-override.
        
        Subclass harus override ini atau implement create_module_ui_components.
        """
        # Try module-specific method first
        if hasattr(self, 'create_module_ui_components'):
            return self.create_module_ui_components(config, **kwargs)
        
        # Fallback: error jika tidak di-implement
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement create_ui_components() "
            "or create_module_ui_components()"
        )
    
    # === Operation Handler Management ===
    
    def register_operation_handler(self, name: str, handler: Any) -> None:
        """Register operation handler."""
        self._operation_handlers[name] = handler
        self._handlers[f'op_{name}'] = handler
        self.logger.debug(f"📌 Registered operation handler: {name}")
    
    def get_operation_handler(self, name: str) -> Optional[Any]:
        """Get operation handler by name."""
        return self._operation_handlers.get(name)
    
    def execute_operation(self, name: str, *args, **kwargs) -> Any:
        """Execute operation by name."""
        handler = self.get_operation_handler(name)
        if not handler:
            raise ValueError(f"Operation handler not found: {name}")
        
        # Check for execute method
        if hasattr(handler, 'execute'):
            return handler.execute(*args, **kwargs)
        elif hasattr(handler, 'execute_operation'):
            return handler.execute_operation(*args, **kwargs)
        else:
            raise AttributeError(f"Operation handler {name} has no execute method")
    
    # === Service Handler Management ===
    
    def register_service_handler(self, name: str, handler: Any) -> None:
        """Register service handler."""
        self._service_handlers[name] = handler
        self._handlers[f'svc_{name}'] = handler
        self.logger.debug(f"📌 Registered service handler: {name}")
    
    def get_service_handler(self, name: str) -> Optional[Any]:
        """Get service handler by name."""
        return self._service_handlers.get(name)
    
    # === Module Handler Delegation ===
    
    @property
    def module_handler(self) -> Optional[ModuleUIHandler]:
        """Get module handler."""
        return self._module_handler
    
    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Update UI dari config via handler."""
        if self._module_handler:
            self._module_handler.update_ui_from_config(config or self.config)
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract config dari UI via handler."""
        if self._module_handler:
            return self._module_handler.extract_config_from_ui()
        return {}
    
    def sync_config_with_ui(self) -> None:
        """Sync config dengan UI."""
        if self._module_handler:
            self._module_handler.sync_config_with_ui()
    
    def sync_ui_with_config(self) -> None:
        """Sync UI dengan config."""
        if self._module_handler:
            self._module_handler.sync_ui_with_config()
    
    # === Event Handler Registration ===
    
    def register_ui_event(self, 
                         component_name: str, 
                         event_name: str, 
                         handler: Callable) -> bool:
        """Register UI event handler."""
        if self._module_handler:
            return self._module_handler.register_event_handler(
                component_name, event_name, handler
            )
        return False
    
    def register_ui_events(self, 
                          events: Dict[str, List[Tuple[str, Callable]]]) -> Dict[str, bool]:
        """Batch register UI events.
        
        Args:
            events: Dict mapping component_name ke list of (event_name, handler) tuples
            
        Returns:
            Dict mapping component_name ke success status
        """
        results = {}
        
        for comp_name, comp_events in events.items():
            comp_results = []
            for event_name, handler in comp_events:
                success = self.register_ui_event(comp_name, event_name, handler)
                comp_results.append(success)
            
            results[comp_name] = all(comp_results)
        
        return results
    
    # === Lifecycle Methods ===
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup operation handlers
        for handler in self._operation_handlers.values():
            if hasattr(handler, 'cleanup'):
                try:
                    handler.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to cleanup handler: {e}")
        
        # Cleanup service handlers
        for handler in self._service_handlers.values():
            if hasattr(handler, 'cleanup'):
                try:
                    handler.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to cleanup handler: {e}")
        
        # Cleanup module handler
        if self._module_handler and hasattr(self._module_handler, 'cleanup'):
            try:
                self._module_handler.cleanup()
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to cleanup module handler: {e}")
        
        self.logger.info("🧹 Module cleanup complete")
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information summary."""
        info = {
            'module': self.full_module_name,
            'initialized': self.is_initialized,
            'init_duration': self.init_duration,
            'components': len(self._ui_components),
            'handlers': {
                'total': len(self._handlers),
                'operations': len(self._operation_handlers),
                'services': len(self._service_handlers)
            },
            'config': {
                'items': len(self.config),
                'shared': hasattr(self.config_handler, '_shared_manager') and 
                         self.config_handler._shared_manager is not None
            }
        }
        
        # Add handler info jika ada
        if self._module_handler:
            info['handler_info'] = self._module_handler.get_module_info()
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"module='{self.full_module_name}', "
                f"initialized={self._initialized}, "
                f"handlers={len(self._handlers)})")