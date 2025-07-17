"""
File: smartcash/ui/setup/dependency/dependency_uimodule.py
Description: Dependency Module implementation using BaseUIModule mixin pattern.
"""

from typing import Dict, Any, Optional, List

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Dependency module imports
from smartcash.ui.setup.dependency.components.dependency_ui import create_dependency_ui_components
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.configs.dependency_defaults import get_default_dependency_config


class DependencyUIModule(BaseUIModule):
    """
    Dependency Module implementation using BaseUIModule with environment support.
    
    Features:
    - 📦 Package management (install, uninstall, check, update)
    - 🌍 Environment-aware package installation (via BaseUIModule environment support)
    - 📊 Real-time package status tracking
    - 🔄 Enhanced factory-based initialization functions
    - ✅ Full compliance with OPERATION_CHECKLISTS.md requirements
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self):
        """Initialize Dependency UI module with environment support."""
        # Initialize BaseUIModule with environment support enabled
        super().__init__(
            module_name='dependency',
            parent_module='setup',
            enable_environment=True  # Enable environment management features
        )
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Dependency-specific attributes
        self._package_status = {}
        
        self.logger.debug("✅ DependencyUIModule initialized")
    
    def _register_default_operations(self) -> None:
        """Register default operation handlers including dependency-specific operations."""
        # Call parent method to register base operations
        super()._register_default_operations()
        
        # Note: Dynamic button handler registration is now handled by BaseUIModule
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Dependency module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Dependency-specific handlers
        dependency_handlers = {
            'install': self._operation_install_packages,
            'uninstall': self._operation_uninstall_packages,
            'update': self._operation_update_packages,
            'check_status': self._operation_check_status,
            'check': self._operation_check_status
        }
        
        handlers.update(dependency_handlers)
        return handlers
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Dependency module (BaseUIModule requirement)."""
        return get_default_dependency_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> DependencyConfigHandler:
        """Create config handler instance for Dependency module (BaseUIModule requirement)."""
        handler = DependencyConfigHandler(config)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Dependency module (BaseUIModule requirement)."""
        try:
            self.logger.debug("Creating Dependency UI components...")
            ui_components = create_dependency_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
            
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Dependency module with environment detection.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = BaseUIModule.initialize(self)
            
            if success:
                # Set UI components in config handler for extraction
                if self._config_handler and hasattr(self._config_handler, 'set_ui_components'):
                    self._config_handler.set_ui_components(self._ui_components)
                
                # Post-initialization logging (now that operation container is ready)
                self._log_initialization_complete()
                
                # Set global instance for convenience access
                global _dependency_module_instance
                _dependency_module_instance = self
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Dependency module: {e}")
            return False
    
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress display components."""
        try:
            # Ensure progress visibility for dependency operations
            self._ensure_progress_visibility()
            
            # Initialize progress bars if needed
            if hasattr(self, '_ui_components') and self._ui_components:
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'initialize'):
                    progress_tracker.initialize()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to initialize progress display: {e}")
    def ensure_components_ready(self) -> bool:
        """Ensure all required UI components are ready for operations.
        
        Returns:
            bool: True if all components are ready, False otherwise
        """
        required_components = ['operation_container', 'progress_tracker']
        
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.log("⚠️ Komponen UI belum diinisialisasi", 'warning')
            return False
            
        missing = [comp for comp in required_components if comp not in self._ui_components]
        if missing:
            self.log(f"⚠️ Komponen UI yang diperlukan belum tersedia: {', '.join(missing)}", 'warning')
            return False
            
        # Ensure progress tracker is ready
        if not self.ensure_progress_ready():
            self.log("⚠️ Progress tracker belum siap", 'warning')
            return False
            
        return True
        
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container (after it's ready)."""
        try:
            # Ensure components are ready before proceeding
            if not self.ensure_components_ready():
                self.log("⚠️ Beberapa komponen UI belum siap, beberapa fitur mungkin terbatas", 'warning')
            
            # Log environment info if environment support is enabled
            if self.has_environment_support:
                env_type = "Google Colab" if self.is_colab else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                
                # Safely access environment_paths attributes
                if hasattr(self, 'environment_paths') and self.environment_paths is not None:
                    if hasattr(self.environment_paths, 'data_root') and self.environment_paths.data_root:
                        self.log(f"📁 Direktori kerja: {self.environment_paths.data_root}", 'info')
                    else:
                        self.log("ℹ️ Direktori kerja default akan digunakan", 'info')
            
            # Update status panel
            self.log("📊 Status: Siap untuk manajemen paket", 'info')
            
        except Exception as e:
            # Use logger fallback if operation container logging fails
            self.logger.error(f"Gagal mencatat inisialisasi selesai: {e}", exc_info=True)
            self.log(f"⚠️ Terjadi kesalahan saat inisialisasi: {str(e)}", 'error')
    # ==================== OPERATION HANDLERS ====================
    
    def _operation_install_packages(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package installation operation using common wrapper."""
        def validate_packages():
            selected_packages = self._get_selected_packages()
            if not selected_packages:
                return {'valid': False, 'message': "Tidak ada paket yang dipilih untuk diinstal"}
            return {'valid': True, 'packages': selected_packages}
        
        def execute_install():
            validation = validate_packages()
            selected_packages = validation.get('packages', [])
            return self._execute_install_operation(selected_packages)
        
        return self._execute_operation_with_wrapper(
            operation_name="Instalasi Paket",
            operation_func=execute_install,
            button=button,
            validation_func=validate_packages,
            success_message="Instalasi paket berhasil diselesaikan",
            error_message="Kesalahan instalasi paket"
        )
    
    def _operation_uninstall_packages(self, button=None) -> Dict[str, Any]:
        """Handle package uninstallation operation using common wrapper."""
        def validate_packages():
            # Ensure UI components are ready first
            if not self.ensure_components_ready():
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            selected_packages = self._get_selected_packages()
            if not selected_packages:
                return {'valid': False, 'message': "Tidak ada paket yang dipilih untuk diuninstal"}
            return {'valid': True, 'packages': selected_packages}
        
        def execute_uninstall():
            validation = validate_packages()
            selected_packages = validation.get('packages', [])
            self.log(f"📦 Paket yang akan diuninstal: {', '.join(selected_packages)}", 'info')
            return self._execute_uninstall_operation(selected_packages)
        
        return self._execute_operation_with_wrapper(
            operation_name="Uninstal Paket",
            operation_func=execute_uninstall,
            button=button,
            validation_func=validate_packages,
            success_message="Uninstal paket berhasil diselesaikan",
            error_message="Kesalahan uninstal paket"
        )
    
    def _operation_update_packages(self, button=None) -> Dict[str, Any]:
        """Handle package update operation using common wrapper."""
        def validate_packages():
            # Ensure UI components are ready first
            if not self.ensure_components_ready():
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            
            selected_packages = self._get_selected_packages()
            if not selected_packages:
                return {'valid': False, 'message': "Tidak ada paket yang dipilih untuk diperbarui"}
            return {'valid': True, 'packages': selected_packages}
        
        def execute_update():
            validation = validate_packages()
            selected_packages = validation.get('packages', [])
            self.log(f"📦 Paket yang akan diperbarui: {', '.join(selected_packages)}", 'info')
            return self._execute_update_operation(selected_packages)
        
        return self._execute_operation_with_wrapper(
            operation_name="Perbarui Paket",
            operation_func=execute_update,
            button=button,
            validation_func=validate_packages,
            success_message="Perbarui paket berhasil diselesaikan",
            error_message="Kesalahan perbarui paket"
        )
    
    def _operation_check_status(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package status check operation using common wrapper."""
        def validate_components():
            if not self.ensure_components_ready():
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            return {'valid': True}
        
        def execute_check_status():
            result = self._execute_check_status_operation()
            
            # Log status summary if successful
            if result.get('success'):
                status_summary = result.get('summary', {})
                installed = status_summary.get('installed', 0)
                missing = status_summary.get('missing', 0)
                total = status_summary.get('total', 0)
                
                status_msg = f"Status: {installed}/{total} terinstal, {missing} hilang"
                self.log(f"📊 {status_msg}", 'info')
                
                # Update internal package status
                self._package_status = result.get('package_status', {})
                
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name="Cek Status Paket",
            operation_func=execute_check_status,
            button=button,
            validation_func=validate_components,
            success_message="Pemeriksaan status selesai",
            error_message="Kesalahan cek status"
        )
    
    # ==================== OPERATION EXECUTION METHODS ====================
    
    def _execute_install_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute installation operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.install_operation import InstallOperationHandler
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = InstallOperationHandler(
                ui_components=ui_components,
                config={**self.get_current_config(), 'explicit_packages': packages}
            )
            
            # Execute the operation - progress tracking handled by operation handler
            return handler.execute_operation()
            
        except Exception as e:
            self.log(f"Error in install operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_uninstall_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute uninstallation operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.uninstall_operation import UninstallOperationHandler
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = UninstallOperationHandler(
                ui_components=ui_components,
                config={**self.get_current_config(), 'explicit_packages': packages}
            )
            
            # Execute the operation
            return handler.execute_operation()
            
        except Exception as e:
            self.log(f"Error in uninstall operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_update_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute update operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.update_operation import UpdateOperationHandler
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = UpdateOperationHandler(
                ui_components=ui_components,
                config={**self.get_current_config(), 'explicit_packages': packages}
            )
            
            # Execute update operation - progress tracking handled by operation handler
            return handler.execute_operation()
            
        except Exception as e:
            self.log(f"Error in update operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_check_status_operation(self) -> Dict[str, Any]:
        """Execute check status operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.check_operation import CheckStatusOperationHandler
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = CheckStatusOperationHandler(
                ui_components=ui_components,
                config=self.get_current_config()
            )
            
            # Execute the operation
            return handler.execute_operation()
            
        except Exception as e:
            self.log(f"Error in check status operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    # ==================== DEPENDENCY-SPECIFIC METHODS ====================
    
    def _get_selected_packages(self) -> list:
        """Get list of selected packages from UI."""
        selected_packages = []
        
        try:
            # Get package checkboxes from UI components
            if self._ui_components and 'package_checkboxes' in self._ui_components:
                checkboxes = self._ui_components['package_checkboxes']
                for _, checkbox_list in checkboxes.items():
                    for checkbox in checkbox_list:
                        if hasattr(checkbox, 'value') and checkbox.value:  # If checkbox is checked
                            if hasattr(checkbox, 'package_name'):
                                selected_packages.append(checkbox.package_name)
                            else:
                                # Fallback: extract package name from description
                                desc = getattr(checkbox, 'description', '')
                                if desc and '(' in desc:
                                    package_name = desc.split('(')[0].strip()
                                    selected_packages.append(package_name)
            
            # Get custom packages
            if self._ui_components and 'custom_packages' in self._ui_components:
                custom_widget = self._ui_components['custom_packages']
                if hasattr(custom_widget, 'value') and custom_widget.value.strip():
                    custom_packages = [pkg.strip() for pkg in custom_widget.value.split(',') if pkg.strip()]
                    selected_packages.extend(custom_packages)
            
        except Exception as e:
            self.log(f"Failed to get selected packages: {e}", 'warning')
        
        return selected_packages
    
    def get_package_status(self) -> Dict[str, Any]:
        """
        Get current package status information.
        
        Returns:
            Package status dictionary
        """
        try:
            status = {
                'initialized': self._is_initialized,
                'module_name': self.module_name,
                'environment_type': 'colab' if self.is_colab else 'local',
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'package_status': self._package_status,
                'environment_paths': self.environment_paths
            }
            
            return status
            
        except Exception as e:
            return {'error': f'Pemeriksaan status gagal: {str(e)}'}
    


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

# Create the initialize function using enhanced factory pattern
initialize_dependency_ui = EnhancedUIModuleFactory.create_display_function(DependencyUIModule)



# ==================== CONVENIENCE FUNCTIONS ====================

# Global module instance for singleton pattern
_dependency_module_instance: Optional[DependencyUIModule] = None

def create_dependency_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> DependencyUIModule:
    """
    Create a new Dependency UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        DependencyUIModule instance
    """
    module = DependencyUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module

def get_dependency_uimodule() -> Optional[DependencyUIModule]:
    """Get the current Dependency UIModule instance."""
    global _dependency_module_instance
    return _dependency_module_instance

def reset_dependency_uimodule() -> None:
    """Reset the global Dependency UIModule instance."""
    global _dependency_module_instance
    if _dependency_module_instance:
        try:
            _dependency_module_instance.cleanup()
        except:
            pass
    _dependency_module_instance = None