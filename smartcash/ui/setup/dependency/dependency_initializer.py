"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer dengan proper inheritance dan implementasi abstract method
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.footer_container import create_footer_container

from .components.dependency_tabs import create_dependency_tabs
from .handlers.dependency_ui_handler import DependencyUIHandler
from .configs.dependency_defaults import get_default_dependency_config
from .components.dependency_ui import create_dependency_ui_components  # New import

class DependencyInitializer(ModuleInitializer):
    """Initializer untuk dependency module dengan proper structure"""
    
    def __init__(self):
        super().__init__(
            module_name='dependency',
            parent_module='setup',
            handler_class=DependencyUIHandler,
            auto_setup_handlers=True
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration"""
        return get_default_dependency_config()
    
    def initialize(self) -> Dict[str, Any]:
        """🚀 Initialize dependency module - implements abstract method"""
        try:
            self.logger.info("🚀 Memulai inisialisasi dependency module...")
            
            # Load config
            config = self.get_default_config()
            
            # Create UI components
            ui_components = create_dependency_ui_components(config)  # Updated to use new function
            
            # Setup handlers
            self.setup_handlers(ui_components)
            
            # Setup operation handlers
            self.setup_operation_handlers()
            
            self.logger.info("✅ Dependency module berhasil diinisialisasi")
            
            return {
                'success': True,
                'ui_components': ui_components,
                'config': config,
                'module_handler': self._module_handler,
                'operation_handlers': self._operation_handlers
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi dependency module: {e}")
            return {
                'success': False,
                'error': str(e),
                'ui_components': {},
                'config': {}
            }
    
    def setup_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup UI handlers untuk dependency"""
        try:
            # Store UI components
            self._ui_components = ui_components
            
            # Create and setup module handler
            if not self._module_handler:
                self._module_handler = self.create_module_handler()
            
            # Setup module handler with UI components
            self._module_handler.setup(ui_components)
            
            # Connect button handlers
            self._connect_button_handlers()
            
            self.logger.info("✅ Handlers berhasil di-setup")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {e}")
            raise
    
    def _connect_button_handlers(self) -> None:
        """Connect button click handlers"""
        try:
            # Get buttons
            install_btn = self._ui_components.get('install_button')
            check_btn = self._ui_components.get('check_updates_button')
            uninstall_btn = self._ui_components.get('uninstall_button')
            
            # Connect handlers
            if install_btn:
                install_btn.on_click(self._on_install_click)
            
            if check_btn:
                check_btn.on_click(self._on_check_updates_click)
            
            if uninstall_btn:
                uninstall_btn.on_click(self._on_uninstall_click)
                
        except Exception as e:
            self.logger.error(f"❌ Error connecting button handlers: {e}")
    
    def _on_install_click(self, btn):
        """Handler untuk install button"""
        try:
            self.logger.info("📥 Install operation dimulai...")
            
            # Extract packages dari UI
            packages_to_install = self._get_packages_for_operation()
            
            if not packages_to_install:
                self._update_status("⚠️ Tidak ada packages yang dipilih untuk instalasi", "warning")
                return
            
            # Show confirmation dialog
            self._show_confirmation_dialog(
                title="Konfirmasi Instalasi",
                message=f"Apakah Anda yakin ingin menginstall {len(packages_to_install)} packages?",
                packages=packages_to_install,
                operation="install"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam install operation: {e}")
            self._update_status(f"❌ Error: {str(e)}", "error")
    
    def _on_check_updates_click(self, btn):
        """Handler untuk check updates button"""
        try:
            self.logger.info("🔄 Check updates operation dimulai...")
            
            # Get operation handler
            check_handler = self.get_operation_handler('check_status')
            if not check_handler:
                self._update_status("❌ Check status handler tidak tersedia", "error")
                return
            
            # Execute check operation
            self._update_status("🔍 Mengecek status packages...", "info")
            result = check_handler.execute_operation()
            
            if result.get('success'):
                installed_count = result.get('installed', 0)
                not_installed_count = result.get('not_installed', 0)
                self._update_status(
                    f"✅ Check selesai: {installed_count} installed, {not_installed_count} not installed", 
                    "success"
                )
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ Check gagal: {error_msg}", "error")
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam check updates operation: {e}")
            self._update_status(f"❌ Error: {str(e)}", "error")
    
    def _on_uninstall_click(self, btn):
        """Handler untuk uninstall button"""
        try:
            self.logger.info("🗑️ Uninstall operation dimulai...")
            
            # Extract packages dari UI
            packages_to_uninstall = self._get_packages_for_operation()
            
            if not packages_to_uninstall:
                self._update_status("⚠️ Tidak ada packages yang dipilih untuk uninstall", "warning")
                return
            
            # Show confirmation dialog dengan warning
            self._show_confirmation_dialog(
                title="⚠️ Konfirmasi Uninstall",
                message=f"PERINGATAN: Anda akan menghapus {len(packages_to_uninstall)} packages. Ini bisa menyebabkan error jika ada dependencies!",
                packages=packages_to_uninstall,
                operation="uninstall",
                danger_mode=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam uninstall operation: {e}")
            self._update_status(f"❌ Error: {str(e)}", "error")
    
    def setup_operation_handlers(self) -> None:
        """Setup operation handlers untuk package management"""
        try:
            from .operations.install_handler import InstallOperationHandler
            from .operations.update_handler import UpdateOperationHandler
            from .operations.uninstall_handler import UninstallOperationHandler
            from .operations.check_status_handler import CheckStatusOperationHandler
            
            self.register_operation_handler('install', InstallOperationHandler(self._ui_components, self.config))
            self.register_operation_handler('update', UpdateOperationHandler(self._ui_components, self.config))
            self.register_operation_handler('uninstall', UninstallOperationHandler(self._ui_components, self.config))
            self.register_operation_handler('check_status', CheckStatusOperationHandler(self._ui_components, self.config))
            
            self.logger.info("✅ Operation handlers berhasil di-setup")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some operation handlers failed to setup: {e}")
    
    def _get_packages_for_operation(self) -> list:
        """Extract packages yang dipilih dari UI untuk operations"""
        try:
            from .components.package_selector import get_selected_packages, get_custom_packages_text
            
            packages = []
            
            # Get selected packages dari categories
            selected_packages = get_selected_packages(self._ui_components)
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
            self.logger.error(f"❌ Error getting packages for operation: {e}")
            return []
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel - delegate to module handler"""
        if self._module_handler:
            self._module_handler.update_status(message, status_type)
        else:
            # Fallback
            getattr(self.logger, status_type, self.logger.info)(message)
    
    def _show_confirmation_dialog(self, title: str, message: str, packages: list, 
                                operation: str, danger_mode: bool = False) -> None:
        """Show confirmation dialog - delegate to module handler"""
        if self._module_handler:
            # Create detailed message
            detailed_message = f"{message}\n\nPackages:\n"
            for pkg in packages[:10]:
                detailed_message += f"• {pkg}\n"
            if len(packages) > 10:
                detailed_message += f"... dan {len(packages) - 10} packages lainnya"
            
            self._module_handler.show_confirmation(
                title=title,
                message=detailed_message,
                on_confirm=lambda: self._execute_operation(operation, packages),
                on_cancel=lambda: self._update_status("❌ Operasi dibatalkan", "info")
            )
        else:
            # Fallback: execute directly
            self._execute_operation(operation, packages)
    
    def _execute_operation(self, operation: str, packages: list) -> None:
        """Execute package operation"""
        try:
            # Update config dengan current packages
            current_config = self.config.copy()
            
            if operation in ['install', 'update']:
                # For install/update, set packages as selected
                selected = []
                custom = []
                
                for pkg in packages:
                    if any(char in pkg for char in ['>', '<', '=']):
                        custom.append(pkg)
                    else:
                        selected.append(pkg)
                
                current_config['selected_packages'] = selected
                current_config['custom_packages'] = '\n'.join(custom)
            
            # Get operation handler
            if operation == 'install':
                handler = self.get_operation_handler('install')
            elif operation == 'update':
                handler = self.get_operation_handler('update')
            elif operation == 'uninstall':
                handler = self.get_operation_handler('uninstall')
            else:
                self._update_status(f"❌ Unknown operation: {operation}", "error")
                return
            
            if not handler:
                self._update_status(f"❌ Handler untuk {operation} tidak tersedia", "error")
                return
            
            # Update handler config
            handler.config = current_config
            
            # Execute operation
            self._update_status(f"🚀 Menjalankan {operation} untuk {len(packages)} packages...", "info")
            result = handler.execute_operation()
            
            # Handle result
            if result.get('success'):
                success_count = result.get('installed', result.get('updated', result.get('uninstalled', 0)))
                self._update_status(f"✅ {operation.title()} berhasil: {success_count} packages", "success")
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ {operation.title()} gagal: {error_msg}", "error")
            
        except Exception as e:
            self.logger.error(f"❌ Error executing {operation}: {e}")
            self._update_status(f"❌ Error dalam {operation}: {str(e)}", "error")

# Global instance
_dependency_initializer: Optional[DependencyInitializer] = None

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """🚀 Initialize dependency UI - main entry point"""
    global _dependency_initializer
    
    try:
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        # Initialize dengan config
        result = _dependency_initializer.initialize()
        
        return {
            'ui_components': result.get('ui_components', {}),
            'module_handler': result.get('module_handler'),
            'config_handler': result.get('config_handler'),
            'operation_handlers': result.get('operation_handlers', {}),
            'success': result.get('success', False)
        }
        
    except Exception as e:
        print(f"❌ Error initializing dependency UI: {e}")
        return {
            'success': False,
            'error': str(e),
            'ui_components': {},
            'module_handler': None,
            'config_handler': None,
            'operation_handlers': {}
        }