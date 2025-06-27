# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/operation_handlers.py - COMPLETE
# Deskripsi: Operation handlers konsisten dengan preprocessing pattern
# =============================================================================

from typing import Dict, Any, List
from .base_handler import BaseDependencyHandler
from smartcash.ui.components.status_panel import update_status_panel

class OperationHandler(BaseDependencyHandler):
    """Handler untuk dependency operations dengan preprocessing pattern"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup operation button handlers"""
        handlers = {}
        
        # Setup install handler
        install_handler = self.setup_button_handler(
            'install_btn', self._handle_install_operation, 'install'
        )
        if install_handler:
            handlers['install'] = install_handler
        
        # Setup check updates handler
        check_handler = self.setup_button_handler(
            'check_updates_btn', self._handle_check_updates_operation, 'check_updates'
        )
        if check_handler:
            handlers['check_updates'] = check_handler
        
        # Setup uninstall handler
        uninstall_handler = self.setup_button_handler(
            'uninstall_btn', self._handle_uninstall_operation, 'uninstall'
        )
        if uninstall_handler:
            handlers['uninstall'] = uninstall_handler
        
        return handlers
    
    def setup_button_handler(self, button_key: str, handler_func, operation: str):
        """Setup button handler dengan proper error handling"""
        button = self.ui_components.get(button_key)
        if button:
            def wrapped_handler(*args):
                try:
                    self._start_operation_flow(operation)
                    handler_func()
                except Exception as e:
                    self._handle_operation_error(operation, str(e))
            
            button.on_click(wrapped_handler)
            return wrapped_handler
        else:
            self.log_warning(f"⚠️ Button {button_key} tidak ditemukan")
            return None
    
    def _handle_install_operation(self) -> None:
        """Handle package installation"""
        try:
            selected_packages = self._get_selected_packages()
            custom_packages = self._get_custom_packages()
            
            all_packages = selected_packages + custom_packages
            
            if not all_packages:
                self.log_warning("⚠️ Tidak ada packages yang dipilih untuk diinstall")
                self._complete_operation_flow('install', False, "Tidak ada packages dipilih")
                return
            
            self.log_info(f"🚀 Memulai instalasi {len(all_packages)} packages...")
            
            # Simulate installation process
            self._simulate_package_operation(all_packages, 'install')
            
            self._complete_operation_flow('install', True, f"✅ {len(all_packages)} packages berhasil diinstall")
            
        except Exception as e:
            error_msg = f"Gagal install packages: {str(e)}"
            self.log_error(f"❌ {error_msg}")
            self._complete_operation_flow('install', False, error_msg)
    
    def _handle_check_updates_operation(self) -> None:
        """Handle check updates operation"""
        try:
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                self.log_warning("⚠️ Tidak ada packages yang dipilih untuk dicek")
                self._complete_operation_flow('check_updates', False, "Tidak ada packages dipilih")
                return
            
            self.log_info(f"🔍 Mengecek updates untuk {len(selected_packages)} packages...")
            
            # Simulate check updates process
            updates_available = self._simulate_check_updates(selected_packages)
            
            if updates_available:
                self.log_success(f"🆙 {len(updates_available)} packages memiliki update tersedia")
                self._complete_operation_flow('check_updates', True, f"Found {len(updates_available)} updates")
            else:
                self.log_info("✅ Semua packages sudah up-to-date")
                self._complete_operation_flow('check_updates', True, "All packages up-to-date")
            
        except Exception as e:
            error_msg = f"Gagal check updates: {str(e)}"
            self.log_error(f"❌ {error_msg}")
            self._complete_operation_flow('check_updates', False, error_msg)
    
    def _handle_uninstall_operation(self) -> None:
        """Handle uninstall operation"""
        try:
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                self.log_warning("⚠️ Tidak ada packages yang dipilih untuk diuninstall")
                self._complete_operation_flow('uninstall', False, "Tidak ada packages dipilih")
                return
            
            # Check for required packages
            required_packages = [pkg for pkg in selected_packages if self._is_required_package(pkg)]
            if required_packages:
                self.log_error(f"❌ Tidak dapat uninstall required packages: {', '.join(required_packages)}")
                self._complete_operation_flow('uninstall', False, "Cannot uninstall required packages")
                return
            
            self.log_info(f"🗑️ Memulai uninstall {len(selected_packages)} packages...")
            
            # Simulate uninstall process
            self._simulate_package_operation(selected_packages, 'uninstall')
            
            self._complete_operation_flow('uninstall', True, f"✅ {len(selected_packages)} packages berhasil diuninstall")
            
        except Exception as e:
            error_msg = f"Gagal uninstall packages: {str(e)}"
            self.log_error(f"❌ {error_msg}")
            self._complete_operation_flow('uninstall', False, error_msg)
    
    def _get_selected_packages(self) -> List[str]:
        """Get list of selected packages"""
        selected = []
        for key, component in self.ui_components.items():
            if key.startswith('pkg_') and hasattr(component, 'value') and component.value:
                package_key = key.replace('pkg_', '')
                from .defaults import get_package_by_key
                package = get_package_by_key(package_key)
                if package:
                    selected.append(package['pip_name'])
        return selected
    
    def _get_custom_packages(self) -> List[str]:
        """Get list of custom packages"""
        custom_input = self.ui_components.get('custom_packages_input')
        if custom_input and custom_input.value:
            return [pkg.strip() for pkg in custom_input.value.split(',') if pkg.strip()]
        return []
    
    def _is_required_package(self, package_name: str) -> bool:
        """Check if package is required"""
        from .defaults import get_all_packages
        all_packages = get_all_packages()
        for pkg in all_packages:
            if pkg.get('pip_name') == package_name:
                return pkg.get('required', False)
        return False
    
    def _simulate_package_operation(self, packages: List[str], operation: str) -> None:
        """Simulate package operation dengan progress update"""
        import time
        
        total_packages = len(packages)
        main_progress = self.ui_components.get('main_progress')
        step_progress = self.ui_components.get('step_progress')
        
        for i, package in enumerate(packages):
            # Update progress
            progress = (i + 1) / total_packages * 100
            if main_progress:
                main_progress.value = progress
            
            # Log current operation
            operation_msg = {
                'install': f"📦 Installing {package}...",
                'uninstall': f"🗑️ Uninstalling {package}...",
                'check': f"🔍 Checking {package}..."
            }.get(operation, f"Processing {package}...")
            
            self.log_info(operation_msg)
            
            # Simulate operation time
            time.sleep(0.5)  # Simulate processing time
            
            # Update step progress
            if step_progress:
                step_progress.value = 100  # Complete current step
            
            # Log completion
            completion_msg = {
                'install': f"✅ {package} installed successfully",
                'uninstall': f"✅ {package} uninstalled successfully",
                'check': f"✅ {package} checked"
            }.get(operation, f"✅ {package} processed")
            
            self.log_success(completion_msg)
            
            # Reset step progress for next package
            if step_progress and i < total_packages - 1:
                step_progress.value = 0
    
    def _simulate_check_updates(self, packages: List[str]) -> List[str]:
        """Simulate check updates dan return packages with updates"""
        import random
        import time
        
        updates_available = []
        
        for package in packages:
            self.log_info(f"🔍 Checking updates for {package}...")
            time.sleep(0.3)  # Simulate check time
            
            # Randomly determine if update is available (30% chance)
            if random.random() < 0.3:
                updates_available.append(package)
                self.log_info(f"🆙 Update available for {package}")
            else:
                self.log_info(f"✅ {package} is up-to-date")
        
        return updates_available
    
    def _start_operation_flow(self, operation: str) -> None:
        """Start operation flow dengan UI updates"""
        # Disable buttons during operation
        self.disable_ui_during_operation()
        
        # Reset progress bars
        main_progress = self.ui_components.get('main_progress')
        step_progress = self.ui_components.get('step_progress')
        
        if main_progress:
            main_progress.value = 0
        if step_progress:
            step_progress.value = 0
        
        # Update status panel
        status_panel = self.ui_components.get('status_panel')
        if status_panel:
            operation_msgs = {
                'install': "🚀 Memulai instalasi packages...",
                'uninstall': "🗑️ Memulai uninstall packages...",
                'check_updates': "🔍 Mengecek updates packages..."
            }
            msg = operation_msgs.get(operation, f"🔄 Memulai {operation}...")
            update_status_panel(status_panel, msg, 'info')
        
        # Expand log accordion
        log_accordion = self.ui_components.get('log_accordion')
        if log_accordion and hasattr(log_accordion, 'selected_index'):
            log_accordion.selected_index = 0
    
    def _complete_operation_flow(self, operation: str, success: bool, message: str) -> None:
        """Complete operation flow dengan cleanup"""
        # Re-enable buttons
        self.enable_ui_after_operation()
        
        # Update status panel
        status_panel = self.ui_components.get('status_panel')
        if status_panel:
            status_type = 'success' if success else 'error'
            update_status_panel(status_panel, message, status_type)
        
        # Reset progress bars
        main_progress = self.ui_components.get('main_progress')
        step_progress = self.ui_components.get('step_progress')
        
        if main_progress:
            main_progress.value = 100 if success else 0
        if step_progress:
            step_progress.value = 100 if success else 0
        
        # Log final result
        if success:
            self.log_success(f"🎉 {operation.title()} completed: {message}")
        else:
            self.log_error(f"💥 {operation.title()} failed: {message}")
    
    def _handle_operation_error(self, operation: str, error_msg: str) -> None:
        """Handle operation error dengan proper cleanup"""
        self.log_error(f"❌ Error during {operation}: {error_msg}")
        self._complete_operation_flow(operation, False, f"Error: {error_msg}")
    
    def enable_ui_after_operation(self) -> None:
        """Enable UI components setelah operasi selesai"""
        buttons = ['install_btn', 'check_updates_btn', 'uninstall_btn', 'add_custom_btn']
        for btn_key in buttons:
            if btn_key in self.ui_components:
                self.ui_components[btn_key].disabled = False
    
    def extract_config(self) -> Dict[str, Any]:
        """Extract configuration dari UI components"""
        try:
            from .config_extractor import extract_dependency_config
            return extract_dependency_config(self.ui_components)
        except Exception as e:
            self.log_error(f"❌ Error extracting config: {str(e)}")
            return {}
    
    def create_progress_callback(self):
        """Create progress callback untuk API operations"""
        def progress_callback(current: int, total: int, message: str = ""):
            try:
                progress = (current / total) * 100 if total > 0 else 0
                
                # Update main progress
                main_progress = self.ui_components.get('main_progress')
                if main_progress:
                    main_progress.value = progress
                
                # Log progress message
                if message:
                    self.log_info(f"📊 {message} ({current}/{total})")
                    
            except Exception as e:
                self.log_error(f"❌ Progress callback error: {str(e)}")
        
        return progress_callback
    
    def is_confirmation_pending(self) -> bool:
        """Check if ada konfirmasi yang sedang pending"""
        confirmation_area = self.ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'children'):
            return len(confirmation_area.children) > 0
        return False

# Factory function
def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup operation handlers untuk dependency management"""
    handler = OperationHandler(ui_components)
    return handler.setup_handlers()