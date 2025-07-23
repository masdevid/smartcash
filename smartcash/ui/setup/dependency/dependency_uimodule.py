"""
File: smartcash/ui/setup/dependency/dependency_uimodule.py
Description: Dependency Module implementation using BaseUIModule mixin pattern.
"""

import traceback

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
        
        # Lazy initialization flags
        self._initialized = False
        self._ui_components = None
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container', 
            'form_container',
            'action_container',
            'operation_container'
        ]
        
        # Initialize package status
        self._package_status = {}
        
        # Minimal logging for performance
        # self.log_debug("✅ DependencyUIModule instance created")  # Disabled for performance
    
    def _register_default_operations(self) -> None:
        """Register default operation handlers including dependency-specific operations."""
        # Minimal logging during registration for performance
        # Call parent method to register base operations
        super()._register_default_operations()
        
        # Reduced debug logging for performance optimization
        # Only log essential information
        if hasattr(self, '_button_handlers') and len(self._button_handlers) > 0:
            pass  # Button handlers registered successfully
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Dependency module-specific button handlers."""
        # Optimized handler registration with minimal logging
        base_handlers = super()._get_module_button_handlers()
        
        # Add Dependency-specific handlers
        dependency_handlers = {
            'install': self._operation_install_packages,
            'uninstall': self._operation_uninstall_packages,
            'update': self._operation_update_packages,
            'check_status': self._operation_check_status
        }
        
        # Merge with base handlers and return
        return {**base_handlers, **dependency_handlers}
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Dependency module (BaseUIModule requirement)."""
        return get_default_dependency_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> DependencyConfigHandler:
        """Create config handler instance for Dependency module (BaseUIModule requirement)."""
        handler = DependencyConfigHandler(config)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and return UI components for the dependency module.
        
        Args:
            config: Configuration dictionary for the UI components
            
        Returns:
            Dictionary of UI components
        """
        if self._ui_components is None:
            try:
                # Optimized UI creation with minimal logging
                self._ui_components = create_dependency_ui_components(module_config=config)
                
                if not self._ui_components:
                    raise RuntimeError("Failed to create UI components")
                
                # Success - minimal logging for performance
                
            except Exception as e:
                self.log_error(f"Error creating UI components: {str(e)}")
                raise
                
        return self._ui_components
    
    def initialize(self) -> bool:
        """Optimized initialization with minimal logging for performance."""
        try:
            # Fast initialization path
            base_result = super().initialize()
            if not base_result:
                return False
            
            # Environment management is handled by BaseUIModule automatically
            # No additional setup needed for dependency module
            
            return True
            
        except Exception as e:
            # Minimal error logging for performance
            self.log_error(f"Initialization failed: {str(e)}")
            return False
                
    # ==================== OPERATION HANDLERS ====================
    
    def _operation_install_packages(self, button=None) -> Dict[str, Any]:
        """Handle package installation operation using common wrapper."""
        def validate_packages():
            """Validate packages before installation."""
            packages = self._get_selected_packages()
            if not packages:
                # Enhanced error message with debugging info
                error_msg = 'Tidak ada paket yang dipilih untuk diinstal. Pastikan Anda telah mencentang checkbox paket atau memasukkan paket kustom.'
                self.log_error(f"❌ Validation failed: {error_msg}")
                return {'valid': False, 'message': error_msg}
            self.log_info(f"✅ Validation passed: {len(packages)} paket dipilih untuk instalasi")
            return {'valid': True, 'packages': packages}
            
        def execute_install():
            """Execute package installation."""
            packages = self._get_selected_packages()
            self.log(f"🔄 Memulai instalasi {len(packages)} paket...", 'info')
            return self._execute_install_operation(packages)
            
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
            """Validate packages before uninstallation."""
            packages = self._get_selected_packages()
            if not packages:
                # Enhanced error message with debugging info
                error_msg = 'Tidak ada paket yang dipilih untuk dihapus. Pastikan Anda telah mencentang checkbox paket atau memasukkan paket kustom.'
                self.log_error(f"❌ Validation failed: {error_msg}")
                return {'valid': False, 'message': error_msg}
            self.log_info(f"✅ Validation passed: {len(packages)} paket dipilih untuk penghapusan")
            return {'valid': True, 'packages': packages}
            
        def execute_uninstall():
            """Execute package uninstallation."""
            packages = self._get_selected_packages()
            self.log(f"🔄 Memulai penghapusan {len(packages)} paket...", 'info')
            return self._execute_uninstall_operation(packages)
            
        return self._execute_operation_with_wrapper(
            operation_name="Penghapusan Paket",
            operation_func=execute_uninstall,
            button=button,
            validation_func=validate_packages,
            success_message="Penghapusan paket berhasil diselesaikan",
            error_message="Kesalahan penghapusan paket"
        )
    
    def _operation_update_packages(self, button=None) -> Dict[str, Any]:
        """Handle package update operation using common wrapper."""
        
        def validate_packages():
            """Validate packages before update."""
            packages = self._get_selected_packages()
            if not packages:
                # Enhanced error message with debugging info
                error_msg = 'Tidak ada paket yang dipilih untuk diperbarui. Pastikan Anda telah mencentang checkbox paket atau memasukkan paket kustom.'
                self.log_error(f"❌ Validation failed: {error_msg}")
                return {'valid': False, 'message': error_msg}
            self.log_info(f"✅ Validation passed: {len(packages)} paket dipilih untuk pembaruan")
            return {'valid': True, 'packages': packages}
            
        def execute_update():
            """Execute package update."""
            packages = self._get_selected_packages()
            self.log(f"🔄 Memulai pembaruan {len(packages)} paket...", 'info')
            return self._execute_update_operation(packages)
            
        return self._execute_operation_with_wrapper(
            operation_name="Pembaruan Paket",
            operation_func=execute_update,
            button=button,
            validation_func=validate_packages,
            success_message="Pembaruan paket berhasil diselesaikan",
            error_message="Kesalahan pembaruan paket"
        )
    
    def _operation_check_status(self, button=None) -> Dict[str, Any]:
        """Handle package status check operation using common wrapper."""
        
        def validate_components():
            return {'valid': True}
        
        def execute_check_status():
            result = self._execute_check_status_operation()
            if result.get('success') and 'summary' in result:
                summary = result['summary']
                total = summary.get('total', 0)
                installed = summary.get('installed', 0)
                missing = summary.get('missing', 0)
                
                # Create informative success message per TASK.md requirements
                if missing > 0 and installed > 0:
                    result['informative_message'] = f"📊 Status: {installed} paket terinstal, {missing} paket belum terinstal dari total {total} paket"
                elif missing == 0 and total > 0:
                    result['informative_message'] = f"✅ Semua {total} paket sudah terinstal"
                elif total == 0:
                    result['informative_message'] = "ℹ️ Tidak ada paket yang dipilih untuk diperiksa"
                else:
                    result['informative_message'] = f"❌ Semua {total} paket belum terinstal"
                    
                self.log_info(result['informative_message'])
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
            # Get package checkboxes from UI components (fix: use correct key with underscore prefix)
            if self._ui_components and '_package_checkboxes' in self._ui_components:
                checkboxes = self._ui_components['_package_checkboxes']
                for category_key, checkbox_list in checkboxes.items():
                    for checkbox in checkbox_list:
                        if hasattr(checkbox, 'value') and checkbox.value:  # If checkbox is checked
                            if hasattr(checkbox, 'package_name'):
                                selected_packages.append(checkbox.package_name)
                            else:
                                # Fallback: extract package name from description
                                desc = getattr(checkbox, 'description', '')
                                if desc and '(' in desc:
                                    package_name = desc.split('(')[0].strip()
                                else:
                                    # If no parentheses, use the whole description
                                    package_name = desc.strip()
                                if package_name:
                                    selected_packages.append(package_name)
            
            # Get custom packages (fix: use correct key with underscore prefix)
            if self._ui_components and '_custom_packages' in self._ui_components:
                custom_widget = self._ui_components['_custom_packages']
                if hasattr(custom_widget, 'value') and custom_widget.value.strip():
                    # Support both comma and newline separated packages
                    custom_text = custom_widget.value.strip()
                    # Split by both comma and newline
                    custom_packages = []
                    for line in custom_text.split('\n'):
                        line = line.strip()
                        if line:
                            # Also split by comma in case user uses comma separation
                            for pkg in line.split(','):
                                pkg = pkg.strip()
                                if pkg:
                                    custom_packages.append(pkg)
                    selected_packages.extend(custom_packages)
            
        except Exception as e:
            self.log(f"Failed to get selected packages: {e}", 'warning')
            # Add debugging information
            if self._ui_components:
                available_keys = list(self._ui_components.keys())
                self.log(f"Available UI component keys: {available_keys}", 'debug')
            else:
                self.log("UI components are None or empty", 'debug')
        
        # Add debug logging for selected packages
        if selected_packages:
            self.log(f"Found {len(selected_packages)} selected packages: {selected_packages}", 'debug')
        else:
            self.log("No packages selected", 'debug')
        
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

    def _validate_packages(self) -> Dict[str, Any]:
        """
        Validate selected packages before operation.
        
        Returns:
            Validation result dictionary
        """
        try:
            packages = self._get_selected_packages()
            
            if not packages:
                return {
                    'valid': False,
                    'message': 'Tidak ada paket yang dipilih untuk operasi',
                    'packages': []
                }
            
            # Check for valid package names
            invalid_packages = []
            for package in packages:
                if not package or not isinstance(package, str) or package.strip() == '':
                    invalid_packages.append(package)
            
            if invalid_packages:
                return {
                    'valid': False,
                    'message': f'Nama paket tidak valid: {invalid_packages}',
                    'packages': packages,
                    'invalid_packages': invalid_packages
                }
            
            return {
                'valid': True,
                'message': f'Validasi berhasil untuk {len(packages)} paket',
                'packages': packages
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Error validasi paket: {str(e)}',
                'packages': []
            }
    
    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup package status
            if hasattr(self, '_package_status'):
                self._package_status.clear()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Dependency module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Dependency module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion