"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer dengan proper inheritance dan implementasi abstract method
"""

import contextlib
import time
from typing import Dict, Any, Optional
import time
import asyncio

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from .configs.dependency_defaults import get_default_dependency_config
from .handlers.dependency_ui_handler import DependencyUIHandler
from .components.dependency_ui import create_dependency_ui_components
from .operations.factory import OperationHandlerFactory
from .operations.operation_manager import OperationType  # Add OperationType import

class DependencyInitializer(ModuleInitializer):
    """Initializer untuk dependency module dengan proper structure"""
    
    def __init__(self, module_name: str = 'dependency', parent_module: Optional[str] = 'setup'):
        # Initialize with handler class and proper module settings
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            handler_class=DependencyUIHandler,
            auto_setup_handlers=True
        )
        self._ui_components = None
        self._operation_handlers = {}
        self._current_operation = None
        self._current_packages = None
        self.logger.info(f"🛠️ DependencyInitializer dibuat untuk modul: {module_name}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration"""
        return get_default_dependency_config()
    
    def _initialize_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """Implementation of initialization logic.
        
        Returns:
            Dict containing initialization results
        """
        # Extract config from args/kwargs
        config = None
        if args:
            config = args[0]
        elif 'config' in kwargs:
            config = kwargs['config']
        
        if config is None:
            config = self.get_default_config()
        try:
            self.logger.info("🚀 Memulai inisialisasi modul dependensi...")
            
            # Pre-initialization phase
            with self._log_step("Pre-initialization checks"):
                self.pre_initialize_checks()
            
            # Configuration phase
            with self._log_step("Load konfigurasi"):
                if config is None:
                    config = self.get_default_config()
            
            # UI components phase
            with self._log_step("Membuat komponen UI"):
                ui_components = create_dependency_ui_components(config)
                if not ui_components:
                    raise RuntimeError("Gagal membuat komponen UI")
                self._ui_components = ui_components
            
            # Handlers setup phase
            with self._log_step("Menyiapkan handlers"):
                self.setup_handlers(ui_components)
            
            # Operation handlers setup phase
            with self._log_step("Menyiapkan operation handlers"):
                self.setup_operation_handlers()
            
            # Post-initialization phase
            with self._log_step("Pembersihan pasca-inisialisasi"):
                self.post_initialize_cleanup()
            
            self._is_initialized = True
            self.logger.info("✅ Modul dependensi berhasil diinisialisasi")
            
            return {
                'success': True,
                'ui_components': self._ui_components,
                'module_handler': self._module_handler,
                'config_handler': self.config_handler,
                'operation_handlers': self._operation_handlers,
                'config': config
            }
            
        except Exception as e:
            from smartcash.ui.core.errors.handlers import get_error_handler
            error_handler = get_error_handler('dependency')
            error_handler.handle_exception(e, 'initialization', fail_fast=False)
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ui_components': {},
                'module_handler': None,
                'config_handler': None,
                'operation_handlers': {}
            }
    
    def pre_initialize_checks(self) -> None:
        """Pre-initialization validation checks"""
        # Check if required imports are available
        try:
            from .components.dependency_tabs import create_dependency_tabs
            from .handlers.dependency_ui_handler import DependencyUIHandler
        except ImportError as e:
            raise RuntimeError(f"Missing required components: {e}")
    
    def post_initialize_cleanup(self) -> None:
        """Post-initialization cleanup and validation"""
        # Validate that essential UI components were created
        if not self._ui_components:
            raise RuntimeError("No UI components were created")
        
        required_components = ['main_container', 'ui']
        missing = [comp for comp in required_components if comp not in self._ui_components]
        if missing:
            self.logger.warning(f"⚠️ Missing components: {missing}")
    
    def setup_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup module handler with UI components.
        
        Args:
            ui_components: Dictionary containing UI components to be managed by the handler.
            
        Raises:
            RuntimeError: If handler setup fails for any reason.
        """
        self.logger.info("🔧 Setting up dependency module handlers...")
        
        # Validate input
        if not ui_components:
            error_msg = "No UI components provided for handler setup"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # Create module handler if it doesn't exist
            if not hasattr(self, '_module_handler') or self._module_handler is None:
                self.logger.debug("Creating new module handler instance")
                self._module_handler = self.create_module_handler()
            
            # Setup handler with UI components
            self.logger.debug("Setting up module handler with UI components")
            self._module_handler.setup(ui_components)
            
            # Initialize handlers dictionary if it doesn't exist
            if not hasattr(self, '_handlers') or not isinstance(self._handlers, dict):
                self._handlers = {}
                
            # Register handlers
            self._handlers.update({
                'module': self._module_handler,
                'config': self._module_handler  # Alias for backward compatibility
            })
            
            self.logger.info(f"✅ Successfully set up {len(self._handlers)} handlers")
            
        except Exception as e:
            error_msg = f"Failed to set up handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    # Removed create_module_handler as it's now handled by the parent class
    
    def _connect_button_handlers(self) -> None:
        """Connect button click handlers to their respective operations.
        
        Raises:
            RuntimeError: If there's an error connecting the handlers
        """
        if not self._ui_components:
            self.logger.warning("No UI components available to connect handlers")
            return
            
        try:
            # Map button names to their operation types
            button_operations = {
                'install_button': OperationType.INSTALL,
                'update_button': OperationType.UPDATE,
                'uninstall_button': OperationType.UNINSTALL,
                'check_status_button': OperationType.CHECK_STATUS
            }
            
            # Connect each button to its operation
            for button_name, operation_type in button_operations.items():
                button = self._ui_components.get(button_name)
                if button and hasattr(button, 'on_click'):
                    button.on_click(lambda _, op=operation_type: self._on_operation_click(op))
            
            self.logger.info("✅ Connected button handlers")
            
        except Exception as e:
            error_msg = f"Failed to connect button handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    async def _execute_operation(self, context: 'OperationContext') -> Dict[str, Any]:
        """Execute package operation using operation manager.
        
        Args:
            context: The operation context
            
        Returns:
            Dict containing operation results
        """
        if not context.packages and context.operation_type != OperationType.CHECK_STATUS:
            self.logger.warning("No packages provided for operation")
            return {'success': False, 'error': 'No packages provided'}
            
        try:
            # Execute the operation asynchronously
            return await self._operation_manager.execute_operation(context)
            
        except Exception as e:
            error_msg = f"❌ Error during {context.operation_type.name.lower()}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.update_status(error_msg, "error")
            return {'success': False, 'error': str(e)}

    def _on_operation_click(self, operation_type: OperationType) -> None:
        """Handle operation button click.
        
        Args:
            operation_type: The type of operation to perform
        """
        try:
            # Get selected packages (except for check_status which can run without selection)
            packages = []
            if operation_type != OperationType.CHECK_STATUS:
                packages = self._get_packages_for_operation()
                if not packages:
                    self.update_status("⚠️ No packages selected", "warning")
                    return
            
            # Create operation context
            context = self._operation_manager.create_operation_context(
                operation_type=operation_type,
                packages=packages,
                status_callback=self.update_status
            )
            
            # Show confirmation for destructive operations
            if operation_type in [OperationType.UNINSTALL, OperationType.UPDATE]:
                self._show_confirmation_dialog(context)
            else:
                # Execute operation directly
                asyncio.create_task(self._execute_operation(context))
                
        except Exception as e:
            error_msg = f"❌ Error during {operation_type.name.lower()}: {str(e)}"
            self.update_status(error_msg, "error")
            self.error_handler.handle_exception(
                e, 
                f'handling {operation_type.name.lower()} click', 
                fail_fast=False,
                extra_context={'operation': operation_type.name.lower()}
            )
            self.update_status(f"❌ Error: {str(e)}", "error")
    
    def setup_operation_handlers(self) -> None:
        """Setup operation handlers untuk package management"""
        try:
            # Create operation handlers
            self._operation_handlers = {
                'install': OperationHandlerFactory.create_handler('install', self._ui_components, self.config),
                'update': OperationHandlerFactory.create_handler('update', self._ui_components, self.config),
                'uninstall': OperationHandlerFactory.create_handler('uninstall', self._ui_components, self.config),
                'check_status': OperationHandlerFactory.create_handler('check_status', self._ui_components, self.config)
            }
            
            self.logger.info("✅ Operation handlers berhasil di-setup")
            
        except Exception as e:
            from smartcash.ui.core.errors.handlers import get_error_handler
            error_handler = get_error_handler('dependency')
            error_handler.handle_exception(e, 'setting up operation handlers', fail_fast=False)

    def _get_packages_for_operation(self) -> list:
        """Extract packages yang dipilih dari UI untuk operations.
        
        Returns:
            List of selected package names/requirements
        """
        if not self._ui_components:
            self.logger.warning("No UI components available to get packages from")
            return []
            
        try:
            from .components.package_selector import get_selected_packages, get_custom_packages_text
            
            packages = []
            
            # Get selected packages dari categories
            selected_packages = get_selected_packages(self._ui_components)
            if selected_packages:
                packages.extend(selected_packages)
            
            # Get custom packages
            custom_packages_text = get_custom_packages_text(self._ui_components)
            if custom_packages_text:
                for line in custom_packages_text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        packages.append(line)
            
            return list(set(packages))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error getting packages for operation: {str(e)}", exc_info=True)
            self.error_handler.handle_exception(
                e, 
                'getting packages for operation', 
                fail_fast=False
            )
            return []

# Global instance for backward compatibility
_dependency_initializer: Optional[DependencyInitializer] = None

def initialize_dependency_ui_internal(config: Optional[Dict[str, Any]] = None) -> Any:
    """Internal dependency UI initialization that returns components.
    
    This function ensures only one instance of the dependency UI is created.
    
    Args:
        config: Configuration dictionary for initialization.
    
    Returns:
        The main UI container widget or dict of components
    """
    global _dependency_initializer
    
    # If we already have an initialized instance, return its UI
    if _dependency_initializer is not None and hasattr(_dependency_initializer, '_ui_components') and _dependency_initializer._ui_components is not None:
        return _dependency_initializer._ui_components.get('ui')
    elif _dependency_initializer is not None:
        # If _ui_components is None, force reinitialization
        _dependency_initializer = None
    
    # Otherwise, create a new instance using ModuleInitializer
    from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
    
    # Use the centralized initialization
    result = ModuleInitializer.initialize_module_ui(
        module_name='dependency',
        parent_module='setup',
        config=config,
        initializer_class=DependencyInitializer
    )
    
    # Store the initializer instance for future use
    _dependency_initializer = ModuleInitializer.get_module_instance('dependency', 'setup')
    
    return result


# Import the display function creator
from smartcash.ui.core.initializers.display_initializer import create_ui_display_function

# Create the initialize function using the consistent pattern with legacy fallback
initialize_dependency_ui = create_ui_display_function(
    module_name='dependency',
    parent_module='setup',
    legacy_function=initialize_dependency_ui_internal
)