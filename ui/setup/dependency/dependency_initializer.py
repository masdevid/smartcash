"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer dengan proper inheritance dan implementasi abstract method
"""

import contextlib
import time
from typing import Dict, Any, Optional
import time

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from .handlers.dependency_ui_handler import DependencyUIHandler
from .components.dependency_ui import create_dependency_ui_components
from .operations.factory import OperationHandlerFactory
from .operations.operation_manager import OperationType  # Add OperationType import

class DependencyInitializer(ModuleInitializer):
    """Initializer untuk dependency module dengan proper structure"""
    
    def __init__(self, module_name: str = 'dependency', parent_module: Optional[str] = 'setup'):
        super().__init__(module_name, parent_module)
        self._ui_components = None
        self._handlers = {}
        self._operation_handlers = {}
        self._module_handler = None
        self._is_initialized = False
        self._confirmation_dialog = None
        self._current_operation = None
        self._current_packages = None
        self.logger.info(f"🛠️ DependencyInitializer dibuat untuk modul: {module_name}")
        
        # Initialize confirmation dialog
        self._init_confirmation_dialog()
        
        # Override the handler instantiation to ensure correct arguments
        if not hasattr(self, '_module_handler') or self._module_handler is None:
            self._module_handler = DependencyUIHandler(module_name='dependency', parent_module='setup')
    
    def _init_confirmation_dialog(self) -> None:
        """Initialize confirmation dialog state.
        
        Note: The actual dialog is now handled by the parent class's show_confirmation method.
        This method is kept for backward compatibility and to maintain the same interface.
        """
        self.logger.debug("Confirmation dialog initialization is now handled by the parent UIHandler")
    
    def __del__(self) -> None:
        """Cleanup resources."""
        # No cleanup needed for confirmation dialog as it's now managed by the parent class
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration"""
        return get_default_dependency_config()
    
    def initialize(self) -> Dict[str, Any]:
        """🚀 Initialize dependency module - implements abstract method"""
        try:
            self.logger.info("🚀 Memulai inisialisasi modul dependensi...")
            
            # Pre-initialization phase
            with self._log_step("Pre-initialization checks"):
                self.pre_initialize_checks()
            
            # Configuration phase
            with self._log_step("Load konfigurasi"):
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
            from smartcash.ui.core.shared.error_handler import get_error_handler
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
    
    def create_module_handler(self) -> DependencyUIHandler:
        """Create module handler instance with correct arguments."""
        return DependencyUIHandler(
            module_name=self.module_name,
            parent_module=self.parent_module
        )
    
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
    
    def _show_confirmation_dialog(self, context: 'OperationContext') -> None:
        """Show confirmation dialog for potentially destructive operations.
        
        Args:
            context: The operation context
        """
        if not context.packages:
            self.logger.warning("No packages provided for confirmation dialog")
            return
            
        # Prepare dialog content
        operation_name = context.operation_type.name.lower()
        title = f"Confirm {operation_name.capitalize()}"
        message = (
            f"Are you sure you want to {operation_name} {len(context.packages)} package(s)?\n\n"
            f"Packages: {', '.join(context.packages[:5])}"
            f"{'...' if len(context.packages) > 5 else ''}"
        )
        
        # Define callbacks
        async def on_confirm():
            self.update_status(
                f"🚀 Starting {operation_name} for {len(context.packages)} packages...", 
                "info"
            )
            await self._execute_operation(context)
            
        def on_cancel():
            self.update_status(f"⚠️ {operation_name.capitalize()} operation cancelled", "info")
        
        # Show confirmation using parent class method
        try:
            self.show_confirmation(
                title=title,
                message=message,
                on_confirm=lambda: asyncio.create_task(on_confirm()),
                on_cancel=on_cancel
            )
        except Exception as e:
            error_msg = f"❌ Error showing confirmation dialog: {str(e)}"
            self.update_status(error_msg, "error")
            self.logger.error(error_msg, exc_info=True)
    
    # Removed duplicate _execute_operation method - using the async version instead

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
            from smartcash.ui.core.shared.error_handler import get_error_handler
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

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Any:
    """🚀 Initialize dependency UI - main entry point
    
    This is a thin wrapper around ModuleInitializer.initialize_module_ui that provides
    backward compatibility with existing code.
    
    Args:
        config (Optional[Dict[str, Any]]): Configuration dictionary for initialization.
    
    Returns:
        Any: The main UI container widget to be displayed.
    """
    from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
    
    # Use the centralized initialization
    return ModuleInitializer.initialize_module_ui(
        module_name='dependency',
        parent_module='setup',
        config=config,
        initializer_class=DependencyInitializer
    )