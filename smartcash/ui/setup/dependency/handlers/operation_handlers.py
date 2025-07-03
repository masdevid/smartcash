"""
File: smartcash/ui/setup/dependency/handlers/operation_handlers.py
Description: Operation handlers for dependency management with centralized error handling and dual progress tracking.

This module provides handlers for package operations (install, uninstall, check updates) with:
- Centralized error handling through BaseDependencyHandler
- Dual progress tracking (overall and per-package progress)
- UI component management during operations
- Integration with summary container for operation results
"""

from typing import Dict, Any, List, Optional, Callable
from .base_dependency_handler import BaseDependencyHandler
from .summary_handler import get_summary_handler

class OperationHandler(BaseDependencyHandler):
    """Handler for dependency operations with centralized error handling and dual progress tracking.
    
    This handler manages package operations (install, uninstall, check updates) with:
    - Centralized error handling through BaseDependencyHandler
    - Dual progress tracking (overall and per-package progress)
    - UI component management during operations
    - Consistent logging and error reporting
    """
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup operation button handlers with centralized error handling.
        
        Returns:
            Dictionary mapping operation names to handler functions
        """
        handlers = {}
        
        # Setup install handler
        install_handler = self.setup_button_handler(
            'install_button', self._handle_install_operation, 'install'
        )
        if install_handler:
            handlers['install'] = install_handler
        
        # Setup check updates handler
        check_handler = self.setup_button_handler(
            'check_button', self._handle_check_updates_operation, 'check_updates'
        )
        if check_handler:
            handlers['check_updates'] = check_handler
        
        # Setup uninstall handler
        uninstall_handler = self.setup_button_handler(
            'uninstall_button', self._handle_uninstall_operation, 'uninstall'
        )
        if uninstall_handler:
            handlers['uninstall'] = uninstall_handler
        
        return handlers
    
    def setup_button_handler(self, button_key: str, handler_func: Callable, operation: str) -> Optional[Callable]:
        """Setup button handler with centralized error handling and UI state management.
        
        Creates a wrapper function that:
        1. Disables UI components during operation
        2. Executes the handler function with proper error handling
        3. Re-enables UI components after operation
        4. Uses dual progress tracking for operation feedback
        
        Args:
            button_key: Key for the button in ui_components
            handler_func: Function to handle the button click
            operation: Operation name for logging and progress tracking
            
        Returns:
            Wrapped handler function or None if button not found
        """
        try:
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
                self.logger.warning(f"⚠️ Button {button_key} tidak ditemukan")
                return None
        except Exception as e:
            self.handle_error(e, f"Failed to setup button handler for {button_key}")
    
    def _handle_install_operation(self) -> None:
        """Handle package installation with centralized error handling"""
        try:
            self.install_packages()
        except Exception as e:
            error_msg = "Gagal install packages"
            self.handle_error(e, error_msg)
            self._complete_operation_flow('install', False, f"{error_msg}: {str(e)}")
    
    def install_packages(self, packages: List[str] = None) -> None:
        """Install selected packages with centralized error handling and dual progress tracking.
        
        Args:
            packages: Optional list of packages to install. If None, uses selected packages.
        """
        try:
            # Get packages to install
            packages_to_install = packages if packages is not None else self._get_selected_packages()
            custom_packages = self._get_custom_packages()
            all_packages = packages_to_install + custom_packages
            
            if not all_packages:
                self.logger.warning("No packages selected for installation")
                summary_handler = get_summary_handler(self.ui_components)
                summary_handler.show_error("Tidak ada paket yang dipilih untuk diinstall")
                return
                
            # Start operation flow
            self._start_operation_flow('install')
            
            # Perform real package installation
            result = self._perform_package_operation(all_packages, 'install')
            
            # Get installation results
            installed_count = len(result.get('success', []))
            failed_count = len(result.get('failed', []))
            success = installed_count > 0 and failed_count == 0
            
            # Create detailed message
            details = ""
            if failed_count > 0 and 'failed' in result:
                details = "Paket yang gagal: " + ", ".join(result['failed'])
            
            # Show installation summary in summary container
            summary_handler = get_summary_handler(self.ui_components)
            summary_handler.show_install_summary(success, installed_count, failed_count, details)
            
            # Complete operation flow
            success_msg = f"Berhasil menginstall {installed_count} paket"
            if failed_count > 0:
                success_msg += f", {failed_count} paket gagal diinstall"
            self._complete_operation_flow('install', success, success_msg)
        except Exception as e:
            self.handle_error(e, "Error installing packages")
            self._complete_operation_flow('install', False, str(e))
    
    def _handle_check_updates_operation(self) -> None:
        """Handle check updates operation with centralized error handling"""
        try:
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                self.logger.warning("⚠️ Tidak ada packages yang dipilih untuk dicek")
                summary_handler = get_summary_handler(self.ui_components)
                summary_handler.show_error("Tidak ada packages yang dipilih untuk dicek")
                self._complete_operation_flow('check_updates', False, "Tidak ada packages dipilih")
                return
            
            self.logger.info(f"🔍 Mengecek updates untuk {len(selected_packages)} packages...")
            
            # Start operation flow
            self._start_operation_flow('check_updates')
            
            # Perform real check updates
            updates_available = self._check_package_updates(selected_packages)
            
            # Show check updates summary in summary container
            summary_handler = get_summary_handler(self.ui_components)
            
            if updates_available:
                # Create detailed message with available updates
                details = "Paket yang memiliki update: " + ", ".join(updates_available)
                msg = f"{len(updates_available)} packages memiliki update tersedia"
                
                # Show update check summary
                summary_handler.show_check_summary(True, len(updates_available), details)
                
                self.logger.warning(f"🔜 {msg}")
                self._complete_operation_flow('check_updates', True, msg)
            else:
                msg = "Semua packages sudah versi terbaru"
                
                # Show update check summary
                summary_handler.show_check_summary(True, 0, "")
                
                self.logger.info(f"✅ {msg}")
                self._complete_operation_flow('check_updates', True, msg)
        except Exception as e:
            error_msg = "Gagal mengecek updates"
            self.handle_error(e, error_msg)
            self._complete_operation_flow('check_updates', False, f"{error_msg}: {str(e)}")
    
    def _handle_uninstall_operation(self) -> None:
        """Handle uninstall operation with centralized error handling"""
        try:
            selected_packages = self._get_selected_packages()
            
            if not selected_packages:
                self.logger.warning("⚠️ Tidak ada packages yang dipilih untuk diuninstall")
                summary_handler = get_summary_handler(self.ui_components)
                summary_handler.show_error("Tidak ada packages yang dipilih untuk diuninstall")
                self._complete_operation_flow('uninstall', False, "Tidak ada packages dipilih")
                return
            
            self.logger.info(f"🗑️ Memulai uninstall {len(selected_packages)} packages...")
            
            # Start operation flow
            self._start_operation_flow('uninstall')
            
            # Perform real uninstallation
            result = self._perform_package_operation(selected_packages, 'uninstall')
            
            # Get uninstallation results
            uninstalled_count = len(result.get('success', []))
            failed_count = len(result.get('failed', []))
            success = uninstalled_count > 0 and failed_count == 0
            
            # Create detailed message
            details = ""
            if failed_count > 0 and 'failed' in result:
                details = "Paket yang gagal diuninstall: " + ", ".join(result['failed'])
            
            # Show uninstallation summary in summary container
            summary_handler = get_summary_handler(self.ui_components)
            summary_handler.show_uninstall_summary(success, uninstalled_count, failed_count, details)
            
            # Complete operation flow
            success_msg = f"Berhasil menghapus {uninstalled_count} paket"
            if failed_count > 0:
                success_msg += f", {failed_count} paket gagal dihapus"
            self._complete_operation_flow('uninstall', success, success_msg)
            
        except Exception as e:
            error_msg = "Gagal uninstall packages"
            self.handle_error(e, error_msg)
            self._complete_operation_flow('uninstall', False, f"{error_msg}: {str(e)}")
    
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
    
    def _perform_package_operation(self, packages: List[str], operation: str, progress_callback=None) -> None:
        """Perform real package operation with dual progress tracking (overall and per-package)
        
        Args:
            packages: List of packages to process
            operation: Type of operation (install, uninstall)
            progress_callback: Optional callback function for progress updates
        """
        try:
            from ..utils import PackageManager
            
            if not packages:
                self.logger.warning(f"No packages selected for {operation}")
                return
                
            total_packages = len(packages)
            self.logger.info(f"Starting {operation} for {total_packages} packages")
            
            # Setup progress with proper total steps
            self.setup_progress(total_packages, f"{operation.title()}")
            
            # Create package manager
            package_manager = PackageManager()
            
            # Process each package
            for i, package in enumerate(packages):
                try:
                    # Get operation emoji for logging
                    operation_emoji = {
                        'install': '💾',
                        'uninstall': '🗑️',
                    }.get(operation, '📦')
                    
                    # Log package operation
                    self.logger.info(f"{operation_emoji} {operation.title()} package: {package}")
                    
                    # Create a progress callback for this package
                    def package_progress_callback(current, total, status):
                        # Calculate progress values
                        step_progress_value = int((current / total) * 100) if total > 0 else 0
                        
                        # Update both progress bars with one call using dual progress tracking
                        self.update_dual_progress(
                            main_value=i+1,  # Main progress is package count
                            step_value=step_progress_value,  # Step progress is percentage within package
                            main_desc=f"{operation.title()} ({i+1}/{total_packages})",
                            step_desc=status
                        )
                    
                    # Perform the actual operation
                    if operation == 'install':
                        success, message = package_manager.install_package(
                            package, 
                            progress_callback=package_progress_callback
                        )
                    elif operation == 'uninstall':
                        success, message = package_manager.uninstall_package(
                            package, 
                            progress_callback=package_progress_callback
                        )
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                    
                    # Log result
                    if success:
                        self.logger.info(f"✅ {operation.title()} of {package} completed successfully")
                    else:
                        self.logger.error(f"❌ {operation.title()} of {package} failed: {message}")
                        
                except Exception as e:
                    self.handle_error(e, f"Error processing package {package}")
                    
                # Reset step progress for next package if not the last one
                if i < total_packages - 1:
                    self.update_dual_progress(
                        main_value=i+1,
                        step_value=0,
                        main_desc=f"{operation.title()} ({i+1}/{total_packages})",
                        step_desc="Preparing next package..."
                    )
        except Exception as e:
            self.handle_error(e, f"Error during {operation} operation: {str(e)}")
    
    def _check_package_updates(self, packages: List[str]) -> List[str]:
        """Check for package updates using real version checking with dual progress tracking.
        
        Uses dual progress tracking to show both overall progress and per-package progress.
        
        Args:
            packages: List of package names to check for updates
            
        Returns:
            List of packages with updates available
        """
        try:
            from ..utils import VersionChecker
            
            if not packages:
                self.logger.warning("No packages selected for update check")
                return []
                
            total_packages = len(packages)
            self.logger.info(f"Checking updates for {total_packages} packages")
            
            # Setup progress with proper total steps
            self.setup_progress(total_packages, "Checking Updates")
            
            # Create version checker
            version_checker = VersionChecker()
            
            # Packages with updates
            updates_available = []
            
            # Process each package
            for i, package in enumerate(packages):
                try:
                    # Log package operation
                    self.logger.info(f"🔍 Checking updates for: {package}")
                    
                    # Create a progress callback for this package
                    def progress_callback(status):
                        # Update both progress bars with one call using dual progress tracking
                        self.update_dual_progress(
                            main_value=i+1,  # Main progress is package count
                            step_value=50,  # Step progress is indeterminate during version check
                            main_desc=f"Checking Updates ({i+1}/{total_packages})",
                            step_desc=status
                        )
                    
                    # Check package status with real version checking
                    package_status = version_checker.check_package_status(
                        package, 
                        progress_callback=progress_callback
                    )
                    
                    # Check if update is available
                    if package_status['update_available']:
                        updates_available.append(package)
                        self.logger.warning(f"🔜 Update available for {package}: {package_status['installed_version']} → {package_status['latest_version']}")
                    else:
                        if package_status['installed']:
                            self.logger.info(f"✓ {package} is up to date ({package_status['installed_version']})")
                        else:
                            self.logger.warning(f"⚠️ {package} is not installed")
                        
                except Exception as e:
                    self.handle_error(e, f"Error checking updates for {package}: {str(e)}")
                    
                # Reset step progress for next package if not the last one
                if i < total_packages - 1:
                    self.update_dual_progress(
                        main_value=i+1,
                        step_value=0,
                        main_desc=f"Checking Updates ({i+1}/{total_packages})",
                        step_desc="Preparing next package..."
                    )
            
            return updates_available
        except Exception as e:
            self.handle_error(e, f"Error during update check operation: {str(e)}")
            return []
    
    def _start_operation_flow(self, operation: str) -> None:
        """Start operation flow with UI updates using centralized error handling and dual progress tracking.
        
        This method prepares the UI for an operation by:
        1. Disabling UI components during operation
        2. Setting up dual progress tracking (overall and per-step)
        3. Clearing previous log entries
        4. Showing loading state in summary container
        
        Args:
            operation: Type of operation being performed (install, uninstall, check_updates)
        """
        try:
            # Use parent class method to disable UI components during operation
            self.disable_ui_during_operation()
            
            # Use parent class method to setup dual progress tracking
            total_steps = 10  # Default value, will be updated during operation
            self.setup_progress(total_steps, operation.title())
            
            # Clear and open log accordion
            self.clear_log_accordion()
            
            # Show loading state in summary container
            summary_handler = get_summary_handler(self.ui_components)
            summary_handler.show_loading(operation)
            
            # Log operation start
            self.logger.info(f"🚀 {operation.title()} operation started")
        except Exception as e:
            self.handle_error(e, f"Error starting operation flow for {operation}")
    
    def _complete_operation_flow(self, operation: str, success: bool, message: str) -> None:
        """Complete operation flow with UI updates using centralized error handling and dual progress tracking.
        
        This method finalizes an operation by:
        1. Re-enabling UI components
        2. Updating status panel with operation result
        3. Completing dual progress tracking with success/failure state
        4. Showing operation result in summary container
        5. Logging the final operation result
        
        Args:
            operation: Type of operation being completed (install, uninstall, check_updates)
            success: Whether the operation was successful
            message: Message to display in status panel and logs
        """
        try:
            # Use parent class method to re-enable UI components
            self.enable_ui_after_operation()
            
            # Update status panel using parent class method
            status_type = 'success' if success else 'error'
            self.update_status_panel(self.ui_components, message, status_type=status_type)
            
            # Use parent class method to complete progress bars
            self.complete_progress(success)
            
            # Show operation result in summary container
            summary_handler = get_summary_handler(self.ui_components)
            summary_handler.show_operation_summary(operation, success, message)
            
            # Log final result
            if success:
                self.logger.info(f"🎉 {operation.title()} completed: {message}")
            else:
                self.logger.error(f"💥 {operation.title()} failed: {message}")
        except Exception as e:
            self.handle_error(e, f"Error completing operation flow for {operation}")
    
    def _handle_operation_error(self, operation: str, error_msg: str) -> None:
        """Handle operation error with proper cleanup using centralized error handling.
        
        This method ensures proper error handling by:
        1. Logging the error with appropriate level and context
        2. Showing error in summary container
        3. Completing the operation flow with failure state
        4. Using centralized error handling for consistent behavior
        
        Args:
            operation: Type of operation that failed (install, uninstall, check_updates)
            error_msg: Detailed error message for logging and display
        """
        try:
            # Log the error
            self.logger.error(f"❌ Error during {operation}: {error_msg}")
            
            # Show error in summary container directly (in case _complete_operation_flow fails)
            try:
                summary_handler = get_summary_handler(self.ui_components)
                summary_handler.show_error(error_msg)
            except Exception as inner_e:
                self.logger.error(f"Failed to show error in summary container: {str(inner_e)}")
                
            # Complete operation flow with failure
            self._complete_operation_flow(operation, False, f"Error: {error_msg}")
        except Exception as e:
            self.handle_error(e, "Error handling operation error")
    
    # Using parent class implementation of enable_ui_after_operation
    
    def extract_config(self) -> Dict[str, Any]:
        """Extract configuration from UI components with centralized error handling.
        
        Returns:
            Dictionary containing extracted configuration or empty dict on error
        """
        try:
            from .config_extractor import extract_dependency_config
            return extract_dependency_config(self.ui_components)
        except Exception as e:
            self.log_error(f"❌ Error extracting config: {str(e)}")
            return {}
    
    def create_progress_callback(self):
        """Create progress callback function for operations with dual progress tracking.
        
        Creates a callback function that:
        1. Updates main progress bar using parent class methods
        2. Updates status panel with operation status
        3. Uses centralized error handling for robustness
        
        Returns:
            Callback function for progress updates
        """
        
        def progress_callback(current, total, status):
            try:
                # Use parent class method for main progress updates
                progress_value = int((current / total) * 100) if total > 0 else 0
                self.update_main_progress(progress_value, f"Progress ({current}/{total})")
                
                # Update status panel using parent class method
                self.update_status_panel(self.ui_components, status, status_type='info')
            except Exception as e:
                self.handle_error(e, "Error updating progress")
        
        return progress_callback
    


# Factory function
def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup operation handlers untuk dependency management"""
    handler = OperationHandler(ui_components)
    return handler.setup_handlers()