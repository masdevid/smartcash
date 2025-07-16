"""
File: smartcash/ui/setup/dependency/dependency_uimodule.py
Description: Dependency Module implementation using BaseUIModule mixin pattern.
"""

from typing import Dict, Any, Optional, List
import sys
import os

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.logger import get_module_logger

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
            # Setup environment management first
            self._setup_environment()
            
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
    
    def _setup_environment(self) -> None:
        """Setup environment management.
        
        This method is called during initialization when environment support is enabled.
        It's automatically handled by BaseUIModule when enable_environment=True.
        """
        # Environment setup is handled by BaseUIModule when enable_environment=True
        # This method is kept for backward compatibility
        if not self.has_environment_support:
            self.logger.warning("Environment support is not enabled")
            return
    
    
    # Button connection is handled automatically by ButtonHandlerMixin in BaseUIModule
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container (after it's ready)."""
        try:
            # Log environment info if environment support is enabled
            if self.has_environment_support:
                env_type = "Google Colab" if self.is_colab else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                if self.environment_paths and 'data_root' in self.environment_paths:
                    self.log(f"📁 Direktori kerja: {self.environment_paths['data_root']}", 'info')
            
            # Update status panel
            self.log("📊 Status: Siap untuk manajemen paket", 'info')
            self.update_operation_status("Siap untuk manajemen paket", "info")
            
        except Exception as e:
            # Use logger fallback if operation container logging fails
            self.logger.debug(f"Post-initialization logging failed: {e}")
    
    # log() and update_operation_status() methods are provided by LoggingMixin and OperationMixin from BaseUIModule
    
    # save_config(), reset_config(), _handle_save_config(), and _handle_reset_config() methods
    # are provided by ConfigurationMixin and BaseUIModule
    
    def start_progress(self, message: str, total: int = 100) -> None:
        """Override start_progress to ensure progress tracker is visible."""
        try:
            # Call parent method which handles initialization
            super().start_progress(message, total)
            
            # Additional logging for dependency module
            self.log(f"📊 Progress started: {message}", 'info')
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Progress tracker initialization failed: {e}")
    
    def update_progress(self, progress: int, message: str = "", level: str = "info") -> None:
        """Override update_progress to ensure progress tracker is visible."""
        try:
            # Call parent method which handles progress updates
            super().update_progress(progress, message, level)
            
            # Additional logging for visibility in dependency module
            if message:
                self.log(f"📊 Progress: {progress}% - {message}", level)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Progress update failed: {e}")
    
    def _register_default_operations(self) -> None:
        """Register default operations for Dependency module."""
        # Call parent method first
        super()._register_default_operations()
        
        # Register Dependency-specific operations
        self.register_operation_handler('install_packages', self._operation_install_packages)
        self.register_operation_handler('uninstall_packages', self._operation_uninstall_packages)
        self.register_operation_handler('check_status', self._operation_check_status)
        self.register_operation_handler('update_packages', self._operation_update_packages)
        self.register_operation_handler('refresh_status', self._operation_refresh_status)
        
        # Register button handlers
        self.register_button_handler('install', self._operation_install_packages)
        self.register_button_handler('uninstall', self._operation_uninstall_packages)
        self.register_button_handler('check_status', self._operation_check_status)
        self.register_button_handler('update', self._operation_update_packages)
    
    # ==================== OPERATION HANDLERS ====================
    
    def _operation_install_packages(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package installation operation using mixin pattern."""
        button_states = None
        try:
            # Start operation logging and progress tracking
            self.log_operation_start("Instalasi Paket")
            self.start_progress("Memulai instalasi paket...", 0)
            self.update_operation_status("Memulai instalasi paket...", "info")
            
            # Disable all buttons during operation
            button_states = self.disable_all_buttons("⏳ Installing packages...")
            
            # Get selected packages from UI
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                warning_msg = "Tidak ada paket yang dipilih untuk diinstal"
                self.log(f"⚠️ {warning_msg}", 'warning')
                self.update_operation_status(warning_msg, "warning")
                self.error_progress(warning_msg)
                return {'success': False, 'message': warning_msg}
            
            # Update progress
            self.update_progress(25, f"Memproses {len(selected_packages)} paket...")
            
            # Execute installation directly using operation handlers
            result = self._execute_install_operation(selected_packages)
            
            if result.get('success'):
                self.log_operation_complete("Instalasi Paket")
                self.update_operation_status("Instalasi paket berhasil diselesaikan", "success")
                installed_count = result.get('installed_count', 0)
                self.log(f"✅ {installed_count} paket berhasil diinstal", 'success')
                self.complete_progress(f"Instalasi selesai: {installed_count} paket berhasil diinstal")
            else:
                error_msg = result.get('message', 'Instalasi gagal')
                self.log_operation_error("Instalasi Paket", error_msg)
                self.update_operation_status(f"Instalasi gagal: {error_msg}", "error")
                self.error_progress(f"Instalasi gagal: {error_msg}")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan instalasi paket: {e}"
            self.log_operation_error("Instalasi Paket", str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable buttons
            if button_states:
                self.enable_all_buttons(button_states)
    
    def _operation_uninstall_packages(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package uninstallation operation using mixin pattern."""
        button_states = None
        try:
            # Start operation logging and progress tracking
            self.log_operation_start("Uninstal Paket")
            self.start_progress("Memulai uninstal paket...", 0)
            self.update_operation_status("Memulai uninstal paket...", "info")
            
            # Disable all buttons during operation
            button_states = self.disable_all_buttons("⏳ Uninstalling packages...")
            
            # Get selected packages from UI
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                warning_msg = "Tidak ada paket yang dipilih untuk diuninstal"
                self.log(f"⚠️ {warning_msg}", 'warning')
                self.update_operation_status(warning_msg, "warning")
                self.error_progress(warning_msg)
                return {'success': False, 'message': warning_msg}
            
            # Update progress
            self.update_progress(25, f"Memproses {len(selected_packages)} paket...")
            
            # Execute uninstallation directly using operation handlers
            result = self._execute_uninstall_operation(selected_packages)
            
            if result.get('success'):
                self.log_operation_complete("Uninstal Paket")
                self.update_operation_status("Uninstal paket berhasil diselesaikan", "success")
                uninstalled_count = result.get('uninstalled_count', 0)
                self.log(f"✅ {uninstalled_count} paket berhasil diuninstal", 'success')
                self.complete_progress(f"Uninstal selesai: {uninstalled_count} paket berhasil diuninstal")
            else:
                error_msg = result.get('message', 'Uninstal gagal')
                self.log_operation_error("Uninstal Paket", error_msg)
                self.update_operation_status(f"Uninstal gagal: {error_msg}", "error")
                self.error_progress(f"Uninstal gagal: {error_msg}")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan uninstal paket: {e}"
            self.log_operation_error("Uninstal Paket", str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable buttons
            if button_states:
                self.enable_all_buttons(button_states)
    
    def _operation_check_status(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package status check operation using mixin pattern."""
        button_states = None
        try:
            # Start operation logging and progress tracking
            self.log_operation_start("Cek Status Paket")
            self.start_progress("Memeriksa status paket...", 0)
            self.update_operation_status("Memeriksa status paket...", "info")
            
            # Disable all buttons during operation
            button_states = self.disable_all_buttons("⏳ Checking package status...")
            
            # Update progress
            self.update_progress(25, "Memproses pemeriksaan status...")
            
            # Execute check status directly using operation handlers
            result = self._execute_check_status_operation()
            
            if result.get('success'):
                self.log_operation_complete("Cek Status Paket")
                self.update_operation_status("Pemeriksaan status selesai", "success")
                
                # Log status summary
                status_summary = result.get('summary', {})
                installed = status_summary.get('installed', 0)
                missing = status_summary.get('missing', 0)
                total = status_summary.get('total', 0)
                
                self.log(f"📊 Status paket: {installed}/{total} terinstal, {missing} hilang", 'info')
                self.complete_progress(f"Pemeriksaan selesai: {installed}/{total} terinstal, {missing} hilang")
                
                # Update internal package status
                self._package_status = result.get('package_status', {})
                
            else:
                error_msg = result.get('message', 'Pemeriksaan status gagal')
                self.log_operation_error("Cek Status Paket", error_msg)
                self.update_operation_status(f"Pemeriksaan gagal: {error_msg}", "error")
                self.error_progress(f"Pemeriksaan gagal: {error_msg}")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan cek status: {e}"
            self.log_operation_error("Cek Status Paket", str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable buttons
            if button_states:
                self.enable_all_buttons(button_states)
    
    def _operation_update_packages(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle package update operation using mixin pattern."""
        button_states = None
        try:
            # Start operation logging and progress tracking
            self.log_operation_start("Update Paket")
            self.start_progress("Memulai update paket...", 0)
            self.update_operation_status("Memulai update paket...", "info")
            
            # Disable all buttons during operation
            button_states = self.disable_all_buttons("⏳ Updating packages...")
            
            # Update progress
            self.update_progress(25, "Memproses update paket...")
            
            # Execute update directly using operation handlers
            result = self._execute_update_operation()
            
            if result.get('success'):
                self.log_operation_complete("Update Paket")
                self.update_operation_status("Update paket berhasil diselesaikan", "success")
                updated_count = result.get('updated_count', 0)
                self.log(f"✅ {updated_count} paket berhasil diupdate", 'success')
                self.complete_progress(f"Update selesai: {updated_count} paket berhasil diupdate")
            else:
                error_msg = result.get('message', 'Update gagal')
                self.log_operation_error("Update Paket", error_msg)
                self.update_operation_status(f"Update gagal: {error_msg}", "error")
                self.error_progress(f"Update gagal: {error_msg}")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan update paket: {e}"
            self.log_operation_error("Update Paket", str(e))
            self.update_operation_status(error_msg, "error")
            self.error_progress(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            # Re-enable buttons
            if button_states:
                self.enable_all_buttons(button_states)
    
    def _operation_refresh_status(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle refresh package status operation using mixin pattern."""
        try:
            self.log_operation_start("Refresh Status")
            self.update_operation_status("Memperbarui status paket...", "info")
            
            # Clear cached status
            self._package_status = {}
            
            # Re-run status check
            result = self._operation_check_status()
            
            if result.get('success'):
                self.log_operation_complete("Refresh Status")
                self.update_operation_status("Status paket diperbarui", "success")
                self.log("🔄 Status paket berhasil diperbarui", 'success')
            
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan refresh status: {e}"
            self.log_operation_error("Refresh Status", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    # ==================== OPERATION EXECUTION METHODS ====================
    
    def _execute_install_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute installation operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.install_operation import InstallOperationHandler
            
            # Update progress
            self.update_progress(50, f"Menginstal {len(packages)} paket...")
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = InstallOperationHandler(
                ui_components=ui_components,
                config={**self.get_current_config(), 'explicit_packages': packages}
            )
            
            # Execute the operation
            result = handler.execute_operation()
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Instalasi selesai")
            else:
                self.update_progress(100, "Instalasi gagal")
                
            return result
            
        except Exception as e:
            self.log(f"Error in install operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_uninstall_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute uninstallation operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.uninstall_operation import UninstallOperationHandler
            
            # Update progress
            self.update_progress(50, f"Menguninstal {len(packages)} paket...")
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = UninstallOperationHandler(
                ui_components=ui_components,
                config={**self.get_current_config(), 'explicit_packages': packages}
            )
            
            # Execute the operation
            result = handler.execute_operation()
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Uninstal selesai")
            else:
                self.update_progress(100, "Uninstal gagal")
                
            return result
            
        except Exception as e:
            self.log(f"Error in uninstall operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_check_status_operation(self) -> Dict[str, Any]:
        """Execute check status operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.check_operation import CheckStatusOperationHandler
            
            # Update progress
            self.update_progress(50, "Memeriksa status paket...")
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = CheckStatusOperationHandler(
                ui_components=ui_components,
                config=self.get_current_config()
            )
            
            # Execute the operation
            result = handler.execute_operation()
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Pemeriksaan selesai")
            else:
                self.update_progress(100, "Pemeriksaan gagal")
                
            return result
            
        except Exception as e:
            self.log(f"Error in check status operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def _execute_update_operation(self) -> Dict[str, Any]:
        """Execute update operation using mixin-based handlers."""
        try:
            from smartcash.ui.setup.dependency.operations.update_operation import UpdateOperationHandler
            
            # Update progress
            self.update_progress(50, "Mengupdate paket...")
            
            # Prepare UI components with operation container for proper logging
            ui_components = self._ui_components.copy()
            ui_components['operation_container'] = self.get_component('operation_container')
            
            # Create handler with current UI components and config
            handler = UpdateOperationHandler(
                ui_components=ui_components,
                config=self.get_current_config()
            )
            
            # Execute the operation
            result = handler.execute_operation()
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Update selesai")
            else:
                self.update_progress(100, "Update gagal")
                
            return result
            
        except Exception as e:
            self.log(f"Error in update operation: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    # Button management methods are provided by ButtonHandlerMixin from BaseUIModule
    
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
                'environment_type': 'colab' if self._environment_manager.is_colab else 'local',
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components),
                'package_status': self._package_status,
                'environment_paths': self._environment_paths
            }
            
            # Add environment-specific information
            if self._environment_manager:
                system_info = self._environment_manager.get_system_info()
                status.update({
                    'python_version': system_info.get('python_version'),
                    'base_directory': system_info.get('base_directory'),
                    'data_directory': system_info.get('data_directory')
                })
            
            return status
            
        except Exception as e:
            return {'error': f'Pemeriksaan status gagal: {str(e)}'}
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get detailed environment information for dependency management."""
        base_info = {
            'is_colab': self._environment_manager.is_colab if self._environment_manager else False,
            'runtime_type': 'colab' if (self._environment_manager and self._environment_manager.is_colab) else 'local',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'paths': self._environment_paths
        }
        
        # Add EnvironmentManager system info if available
        if self._environment_manager:
            try:
                system_info = self._environment_manager.get_system_info()
                base_info.update({
                    'base_directory': system_info.get('base_directory'),
                    'data_directory': system_info.get('data_directory'),
                    'python_executable': sys.executable
                })
            except Exception as e:
                self.logger.warning(f"Failed to get system info from EnvironmentManager: {e}")
        
        return base_info


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

# Create the initialize function using enhanced factory pattern
initialize_dependency_ui = EnhancedUIModuleFactory.create_display_function(DependencyUIModule)

def get_dependency_components(config: Optional[Dict[str, Any]] = None, 
                             **kwargs) -> Optional[Dict[str, Any]]:
    """Get Dependency UI components without displaying."""
    return initialize_dependency_ui(config=config, display=False, **kwargs)


# ==================== CONVENIENCE FUNCTIONS ====================

# Global module instance for singleton pattern
_dependency_module_instance: Optional[DependencyUIModule] = None

def create_dependency_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    enable_environment: bool = True,
    **kwargs
) -> DependencyUIModule:
    """
    Create a new Dependency UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        enable_environment: Whether to enable environment management features
        **kwargs: Additional arguments
        
    Returns:
        DependencyUIModule instance
    """
    module = DependencyUIModule()
    
    if auto_initialize:
        module.initialize(config, enable_environment=enable_environment, **kwargs)
    
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

def display_dependency_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Display Dependency UI and return components."""
    return initialize_dependency_ui(config=config, display=True, **kwargs)


# ==================== SHARED METHODS REGISTRATION ====================

def register_dependency_shared_methods() -> None:
    """Register shared methods for Dependency module (Operation Checklist 9.1)."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register Dependency-specific shared methods
        SharedMethodRegistry.register_method(
            'dependency.get_package_status',
            lambda pkg: get_package_status_for_package(pkg),
            description='Get package installation status'
        )
        
        SharedMethodRegistry.register_method(
            'dependency.list_packages',
            lambda: list_installed_packages(),
            description='List all installed packages'
        )
        
        SharedMethodRegistry.register_method(
            'dependency.get_status',
            lambda: create_dependency_uimodule().get_package_status(),
            description='Get dependency module status'
        )
        
        logger = get_module_logger("smartcash.ui.setup.dependency.shared")
        logger.debug("📋 Registered Dependency shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.setup.dependency.shared")
        logger.error(f"Failed to register shared methods: {e}")


def get_package_status_for_package(package_name: str) -> Dict[str, Any]:
    """Get status of a specific package."""
    try:
        import subprocess
        result = subprocess.run(['pip', 'show', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {"installed": True, "package": package_name, "details": result.stdout}
        else:
            return {"installed": False, "package": package_name}
    except Exception as e:
        return {"installed": False, "package": package_name, "error": str(e)}

def list_installed_packages() -> Dict[str, Any]:
    """List all installed packages."""
    try:
        import subprocess
        result = subprocess.run(['pip', 'list', '--format=json'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            return {"success": True, "packages": packages}
        else:
            return {"success": False, "error": "Failed to list packages"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Auto-register when module is imported
try:
    register_dependency_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")