"""
File: smartcash/ui/setup/dependency/handlers/summary_handler.py
Description: Handler for summary container integration with centralized error handling.

This module provides a handler for managing the summary container in dependency management UI.
It handles displaying operation results (success/failure of install, update, uninstall) in the summary container.
"""

from typing import Dict, Any, Optional
from .base_dependency_handler import BaseDependencyHandler


class SummaryHandler(BaseDependencyHandler):
    """Handler for summary container integration with centralized error handling."""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize summary handler with centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
        
    def show_welcome_message(self) -> None:
        """Show welcome message in summary container."""
        try:
            if 'summary_container' in self.ui_components:
                self.ui_components['summary_container'].set_theme('info')
                self.ui_components['summary_container'].set_content(
                    "<b>Selamat Datang di Dependency Manager</b><br>"
                    "Silakan pilih paket yang ingin diinstall atau diupdate."
                )
                self.logger.debug("Welcome message displayed in summary container")
        except Exception as e:
            self.handle_error(e, "Failed to show welcome message in summary container")
    
    def show_operation_summary(self, operation: str, success: bool, message: str) -> None:
        """Show operation summary in summary container.
        
        Args:
            operation: Type of operation (install, update, uninstall, check)
            success: Whether the operation was successful
            message: Message to display
        """
        try:
            if 'summary_container' not in self.ui_components:
                self.logger.warning("Summary container not found in UI components")
                return
                
            summary_container = self.ui_components['summary_container']
            
            # Set theme based on operation result
            theme = 'success' if success else 'danger'
            summary_container.set_theme(theme)
            
            # Get operation title in Bahasa Indonesia
            operation_titles = {
                'install': 'Instalasi',
                'update': 'Update',
                'uninstall': 'Uninstall',
                'check': 'Pemeriksaan',
            }
            operation_title = operation_titles.get(operation, operation.title())
            
            # Create icon based on operation and result
            icon = '✅' if success else '❌'
            
            # Set content with operation-specific formatting
            title = f"{icon} {operation_title}"
            summary_container.set_content(
                f"<b>{title}</b><br>{message}"
            )
            
            self.logger.debug(f"Operation summary displayed in summary container: {operation}, success={success}")
        except Exception as e:
            self.handle_error(e, "Failed to show operation summary in summary container")
    
    def show_install_summary(self, success: bool, installed_count: int, failed_count: int = 0, details: str = "") -> None:
        """Show installation summary in summary container.
        
        Args:
            success: Whether the installation was successful
            installed_count: Number of packages installed successfully
            failed_count: Number of packages that failed to install
            details: Additional details to display
        """
        try:
            message = f"{installed_count} paket berhasil diinstall."
            if failed_count > 0:
                message += f" {failed_count} paket gagal diinstall."
            
            if details:
                message += f"<br><small>{details}</small>"
                
            self.show_operation_summary('install', success, message)
        except Exception as e:
            self.handle_error(e, "Failed to show installation summary")
    
    def show_update_summary(self, success: bool, updated_count: int, failed_count: int = 0, details: str = "") -> None:
        """Show update summary in summary container.
        
        Args:
            success: Whether the update was successful
            updated_count: Number of packages updated successfully
            failed_count: Number of packages that failed to update
            details: Additional details to display
        """
        try:
            message = f"{updated_count} paket berhasil diupdate."
            if failed_count > 0:
                message += f" {failed_count} paket gagal diupdate."
            
            if details:
                message += f"<br><small>{details}</small>"
                
            self.show_operation_summary('update', success, message)
        except Exception as e:
            self.handle_error(e, "Failed to show update summary")
    
    def show_uninstall_summary(self, success: bool, uninstalled_count: int, failed_count: int = 0, details: str = "") -> None:
        """Show uninstallation summary in summary container.
        
        Args:
            success: Whether the uninstallation was successful
            uninstalled_count: Number of packages uninstalled successfully
            failed_count: Number of packages that failed to uninstall
            details: Additional details to display
        """
        try:
            message = f"{uninstalled_count} paket berhasil diuninstall."
            if failed_count > 0:
                message += f" {failed_count} paket gagal diuninstall."
            
            if details:
                message += f"<br><small>{details}</small>"
                
            self.show_operation_summary('uninstall', success, message)
        except Exception as e:
            self.handle_error(e, "Failed to show uninstallation summary")
    
    def show_check_summary(self, success: bool, update_count: int, details: str = "") -> None:
        """Show check updates summary in summary container.
        
        Args:
            success: Whether the check was successful
            update_count: Number of packages with updates available
            details: Additional details to display
        """
        try:
            if update_count > 0:
                message = f"{update_count} paket memiliki update tersedia."
            else:
                message = "Semua paket sudah versi terbaru."
            
            if details:
                message += f"<br><small>{details}</small>"
                
            self.show_operation_summary('check', success, message)
        except Exception as e:
            self.handle_error(e, "Failed to show check updates summary")
    
    def show_loading(self, operation: str) -> None:
        """Show loading state in summary container.
        
        Args:
            operation: Type of operation being performed
        """
        try:
            if 'summary_container' in self.ui_components:
                # Get operation title in Bahasa Indonesia
                operation_titles = {
                    'install': 'Instalasi',
                    'update': 'Update',
                    'uninstall': 'Uninstall',
                    'check': 'Pemeriksaan',
                }
                operation_title = operation_titles.get(operation, operation.title())
                
                self.ui_components['summary_container'].set_theme('warning')
                self.ui_components['summary_container'].set_content(
                    f"<b>⏳ {operation_title} sedang berjalan...</b><br>"
                    f"Mohon tunggu sebentar."
                )
                self.logger.debug(f"Loading state displayed in summary container for {operation}")
        except Exception as e:
            self.handle_error(e, "Failed to show loading state in summary container")
    
    def show_error(self, message: str) -> None:
        """Show error message in summary container.
        
        Args:
            message: Error message to display
        """
        try:
            if 'summary_container' in self.ui_components:
                self.ui_components['summary_container'].set_theme('danger')
                self.ui_components['summary_container'].set_content(
                    f"<b>❌ Error</b><br>{message}"
                )
                self.logger.debug(f"Error message displayed in summary container: {message}")
        except Exception as e:
            self.handle_error(e, "Failed to show error message in summary container")


# Factory function
def get_summary_handler(ui_components: Dict[str, Any]) -> SummaryHandler:
    """Get summary handler instance.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        SummaryHandler instance
    """
    return SummaryHandler(ui_components=ui_components)
