"""
File: smartcash/ui/setup/dependency/dependency_uimodule.py
Description: Dependency Module implementation using BaseUIModule pattern with operation checklist compliance.
"""

from typing import Dict, Any, Optional
import sys
import os

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.logger import get_module_logger

# Environment management imports
from smartcash.common.environment import get_environment_manager, EnvironmentManager
from smartcash.common.constants.paths import get_paths_for_environment

# Dependency module imports
from smartcash.ui.setup.dependency.components.dependency_ui import create_dependency_ui_components
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.configs.dependency_defaults import get_default_dependency_config
from smartcash.ui.setup.dependency.operations.operation_manager import DependencyOperationManager


class DependencyUIModule(BaseUIModule):
    """
    Dependency Module implementation using BaseUIModule pattern.
    
    Features:
    - 📦 Package management (install, uninstall, check, update)
    - 🌍 Environment-aware package installation
    - 📊 Real-time package status tracking
    - 🔄 Enhanced factory-based initialization functions
    - ✅ Full compliance with OPERATION_CHECKLISTS.md requirements
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self):
        """Initialize Dependency UI module."""
        super().__init__(
            module_name='dependency',
            parent_module='setup'
        )
        
        # Set required components for validation (Operation Checklist 1.2)
        self._required_components = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Dependency-specific attributes
        self._operation_manager: Optional[DependencyOperationManager] = None
        self._environment_manager: Optional[EnvironmentManager] = None
        self._package_status = {}
        self._environment_paths = {}
        
        # Initialize log buffer for pre-operation-container logs
        self._log_buffer = []
        
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
            success = super().initialize()
            
            if success:
                # Setup operation manager after UI components are created
                self._setup_operation_manager()
                
                # Post-initialization logging (now that operation container is ready)
                self._log_initialization_complete()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Dependency module: {e}")
            return False
    
    def _setup_environment(self) -> None:
        """Setup environment management using EnvironmentManager."""
        try:
            # Use standardized environment manager
            self._environment_manager = get_environment_manager(logger=self.logger)
            
            # Get appropriate paths for current environment
            self._environment_paths = get_paths_for_environment(
                is_colab=self._environment_manager.is_colab,
                is_drive_mounted=self._environment_manager.is_drive_mounted if self._environment_manager.is_colab else False
            )
            
            env_type = "Google Colab" if self._environment_manager.is_colab else "Lokal/Jupyter"
            self.logger.debug(f"✅ Environment detected: {env_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup environment: {e}")
            # Create fallback environment info
            self._environment_paths = {
                'data_root': 'data',
                'config': './smartcash/configs'
            }
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager with UI integration."""
        try:
            if not self._ui_components:
                raise RuntimeError("UI components must be created before operation manager")
            
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                raise RuntimeError("Operation container not found in UI components")
            
            self._operation_manager = DependencyOperationManager(
                config=self.get_current_config(),
                operation_container=operation_container,
                ui_components={'operation_container': operation_container},
                environment_manager=self._environment_manager
            )
            
            # Initialize operation manager (note: some operation managers don't have initialize method)
            if hasattr(self._operation_manager, 'initialize'):
                self._operation_manager.initialize()
            else:
                self.logger.debug("Operation manager doesn't have initialize method - using default setup")
            
            self.logger.debug("✅ Operation manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}")
            raise
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container (after it's ready)."""
        try:
            # First, flush any buffered logs
            self._flush_log_buffer()
            
            # Log environment information (Operation Checklist 3.2)
            env_type = "Google Colab" if self._environment_manager.is_colab else "Lokal/Jupyter"
            self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
            self.log(f"📁 Direktori kerja: {self._environment_paths.get('data_root', 'Unknown')}", 'info')
            
            # Update status panel (Operation Checklist 7.1)
            self.update_operation_status("Siap untuk manajemen paket", "info")
            
        except Exception as e:
            # Use logger fallback if operation container logging fails
            self.logger.debug(f"Post-initialization logging failed: {e}")
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not self._log_buffer:
                return
                
            # Display all buffered logs to operation container
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer
            self._log_buffer.clear()
            
        except Exception as e:
            self.logger.debug(f"Failed to flush log buffer: {e}")
    
    def _register_default_operations(self) -> None:
        """Register default operations for Dependency module (Operation Checklist 9.1)."""
        # Call parent method first
        super()._register_default_operations()
        
        # Register Dependency-specific operations (Operation Checklist 8.3)
        self.register_operation_handler('install_packages', self._handle_install_packages)
        self.register_operation_handler('uninstall_packages', self._handle_uninstall_packages)
        self.register_operation_handler('check_status', self._handle_check_status)
        self.register_operation_handler('update_packages', self._handle_update_packages)
        self.register_operation_handler('refresh_status', self._handle_refresh_status)
        
        # Register button handlers (Operation Checklist 2.2)
        self.register_button_handler('install', self._handle_install_packages)
        self.register_button_handler('uninstall', self._handle_uninstall_packages)
        self.register_button_handler('check_status', self._handle_check_status)
        self.register_button_handler('update', self._handle_update_packages)
    
    # ==================== OPERATION HANDLERS ====================
    
    def _handle_install_packages(self, button=None) -> Dict[str, Any]:
        """Handle package installation operation (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Instalasi Paket")
            self.update_operation_status("Memulai instalasi paket...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            # Get selected packages from UI
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                warning_msg = "Tidak ada paket yang dipilih untuk diinstal"
                self.log(f"⚠️ {warning_msg}", 'warning')
                self.update_operation_status(warning_msg, "warning")
                return {'success': False, 'message': warning_msg}
            
            # Execute installation with progress tracking (Operation Checklist 3.1)
            result = self._operation_manager.execute_install(selected_packages)
            
            if result.get('success'):
                self.log_operation_complete("Instalasi Paket")
                self.update_operation_status("Instalasi paket berhasil diselesaikan", "info")
                installed_count = result.get('installed_count', 0)
                self.log(f"✅ {installed_count} paket berhasil diinstal", 'success')
            else:
                error_msg = result.get('message', 'Instalasi gagal')
                self.log_operation_error("Instalasi Paket", error_msg)
                self.update_operation_status(f"Instalasi gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan instalasi paket: {e}"
            self.log_operation_error("Instalasi Paket", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_uninstall_packages(self, button=None) -> Dict[str, Any]:
        """Handle package uninstallation operation (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Uninstal Paket")
            self.update_operation_status("Memulai uninstal paket...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            # Get selected packages from UI
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                warning_msg = "Tidak ada paket yang dipilih untuk diuninstal"
                self.log(f"⚠️ {warning_msg}", 'warning')
                self.update_operation_status(warning_msg, "warning")
                return {'success': False, 'message': warning_msg}
            
            result = self._operation_manager.execute_uninstall(selected_packages)
            
            if result.get('success'):
                self.log_operation_complete("Uninstal Paket")
                self.update_operation_status("Uninstal paket berhasil diselesaikan", "info")
                uninstalled_count = result.get('uninstalled_count', 0)
                self.log(f"✅ {uninstalled_count} paket berhasil diuninstal", 'success')
            else:
                error_msg = result.get('message', 'Uninstal gagal')
                self.log_operation_error("Uninstal Paket", error_msg)
                self.update_operation_status(f"Uninstal gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan uninstal paket: {e}"
            self.log_operation_error("Uninstal Paket", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_check_status(self, button=None) -> Dict[str, Any]:
        """Handle package status check operation (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Cek Status Paket")
            self.update_operation_status("Memeriksa status paket...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            result = self._operation_manager.execute_check_status()
            
            if result.get('success'):
                self.log_operation_complete("Cek Status Paket")
                self.update_operation_status("Pemeriksaan status selesai", "info")
                
                # Log status summary
                status_summary = result.get('summary', {})
                installed = status_summary.get('installed', 0)
                missing = status_summary.get('missing', 0)
                total = status_summary.get('total', 0)
                
                self.log(f"📊 Status paket: {installed}/{total} terinstal, {missing} hilang", 'info')
                
                # Update internal package status
                self._package_status = result.get('package_status', {})
                
            else:
                error_msg = result.get('message', 'Pemeriksaan status gagal')
                self.log_operation_error("Cek Status Paket", error_msg)
                self.update_operation_status(f"Pemeriksaan gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan cek status: {e}"
            self.log_operation_error("Cek Status Paket", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_update_packages(self, button=None) -> Dict[str, Any]:
        """Handle package update operation (Operation Checklist 8.3)."""
        try:
            self.log_operation_start("Update Paket")
            self.update_operation_status("Memulai update paket...", "info")
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not available")
            
            result = self._operation_manager.execute_update()
            
            if result.get('success'):
                self.log_operation_complete("Update Paket")
                self.update_operation_status("Update paket berhasil diselesaikan", "info")
                updated_count = result.get('updated_count', 0)
                self.log(f"✅ {updated_count} paket berhasil diupdate", 'success')
            else:
                error_msg = result.get('message', 'Update gagal')
                self.log_operation_error("Update Paket", error_msg)
                self.update_operation_status(f"Update gagal: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan update paket: {e}"
            self.log_operation_error("Update Paket", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    def _handle_refresh_status(self, button=None) -> Dict[str, Any]:
        """Handle refresh package status operation."""
        try:
            self.log_operation_start("Refresh Status")
            self.update_operation_status("Memperbarui status paket...", "info")
            
            # Clear cached status
            self._package_status = {}
            
            # Re-run status check
            result = self._handle_check_status()
            
            if result.get('success'):
                self.log_operation_complete("Refresh Status")
                self.update_operation_status("Status paket diperbarui", "info")
                self.log("🔄 Status paket berhasil diperbarui", 'success')
            
            return result
            
        except Exception as e:
            error_msg = f"Kesalahan refresh status: {e}"
            self.log_operation_error("Refresh Status", str(e))
            self.update_operation_status(error_msg, "error")
            return {'success': False, 'message': error_msg}
    
    # ==================== DEPENDENCY-SPECIFIC METHODS ====================
    
    def _get_selected_packages(self) -> list:
        """Get list of selected packages from UI."""
        selected_packages = []
        
        try:
            # Get package checkboxes from UI components
            if self._ui_components and 'package_checkboxes' in self._ui_components:
                checkboxes = self._ui_components['package_checkboxes']
                for category, boxes in checkboxes.items():
                    for package_name, checkbox in boxes.items():
                        if checkbox.value:  # If checkbox is checked
                            selected_packages.append(package_name)
            
            # Get custom packages
            if self._ui_components and 'custom_packages' in self._ui_components:
                custom_widget = self._ui_components['custom_packages']
                if hasattr(custom_widget, 'value') and custom_widget.value.strip():
                    custom_packages = [pkg.strip() for pkg in custom_widget.value.split(',') if pkg.strip()]
                    selected_packages.extend(custom_packages)
            
        except Exception as e:
            self.logger.warning(f"Failed to get selected packages: {e}")
        
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
                'operation_manager_ready': self._operation_manager is not None,
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

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None, 
                            display: bool = True, 
                            **kwargs) -> Optional[Dict[str, Any]]:
    """Initialize and optionally display the Dependency UI module."""
    # Filter out conflicting display-related parameters from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['display', 'show_display']}
    
    # Handle the case where display might be in kwargs (parameter conflict resolution)
    final_display = kwargs.get('display', display)
    
    return EnhancedUIModuleFactory.create_and_display(
        module_class=DependencyUIModule,
        config=config,
        display=final_display,
        **filtered_kwargs
    )

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