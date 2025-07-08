"""
File: smartcash/ui/dataset/downloader/operations/manager.py
Operation manager for dataset downloader that extends OperationHandler following colab/dependency pattern
"""

from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .download_operation import DownloadOperationHandler
from .check_operation import CheckOperationHandler
from .cleanup_operation import CleanupOperationHandler

class DownloaderOperationManager(OperationHandler):
    """Operation manager for dataset downloader that extends OperationHandler."""
    
    def __init__(self, config: Dict[str, Any], operation_container=None, **kwargs):
        """Initialize the downloader operation manager."""
        super().__init__(
            module_name='downloader_operation_manager',
            parent_module='downloader',
            operation_container=operation_container,
            **kwargs
        )
        self.config = config
        
        # Initialize operation handlers
        self.download_handler = None
        self.check_handler = None
        self.cleanup_handler = None
    
    def initialize(self) -> None:
        """Initialize the downloader operation manager."""
        self.logger.info("🚀 Initializing Downloader operation manager")
        
        # Initialize operation handlers with UI components
        ui_components = getattr(self, '_ui_components', {})
        if hasattr(self.operation_container, 'get_ui_components'):
            ui_components.update(self.operation_container.get_ui_components())
        
        # Ensure operation container is available
        ui_components['operation_container'] = self.operation_container
        
        self.download_handler = DownloadOperationHandler(ui_components=ui_components)
        self.check_handler = CheckOperationHandler(ui_components=ui_components)
        self.cleanup_handler = CleanupOperationHandler(ui_components=ui_components)
        
        self.logger.info("✅ Downloader operation manager initialization complete")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'download': self.execute_download,
            'check': self.execute_check,
            'cleanup': self.execute_cleanup
        }
    
    async def execute_download(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dataset download operation."""
        try:
            if not self.download_handler:
                self.initialize()
            
            # Execute the actual download
            result = self.download_handler.execute_download(ui_config)
            
            # Update operation summary with results
            await self._update_operation_summary('download', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_download: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_check(self) -> Dict[str, Any]:
        """Execute dataset check operation."""
        try:
            if not self.check_handler:
                self.initialize()
            
            # Execute the actual check
            result = self.check_handler.execute_check()
            
            # Update operation summary with results
            await self._update_operation_summary('check', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_check: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_cleanup(self, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute dataset cleanup operation."""
        try:
            if not self.cleanup_handler:
                self.initialize()
            
            # Get cleanup targets if not provided
            if targets is None:
                targets_result = self.cleanup_handler.get_cleanup_targets()
                if not targets_result.get('success'):
                    return targets_result
                targets = targets_result.get('targets', [])
            
            # Execute the actual cleanup
            result = self.cleanup_handler.execute_cleanup({'targets': targets})
            
            # Update operation summary with results
            await self._update_operation_summary('cleanup', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in execute_cleanup: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _update_operation_summary(self, operation_type: str, result: Dict[str, Any]) -> None:
        """Update operation summary with operation results."""
        try:
            from ..components.operation_summary import update_operation_summary
            
            # Get the operation summary widget from UI components
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            operation_summary = ui_components.get('operation_summary')
            if operation_summary:
                # Determine status type based on result
                if result.get('cancelled'):
                    status_type = 'warning'
                elif result.get('success'):
                    status_type = 'success'
                else:
                    status_type = 'error'
                
                # Update the summary widget
                update_operation_summary(operation_summary, operation_type, result, status_type)
                self.logger.info(f"✅ Updated operation summary for {operation_type}")
            else:
                self.logger.warning("⚠️ Operation summary widget not found in UI components")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update operation summary: {e}")

class DownloadHandlerManager(DownloaderOperationManager):
    """Manager untuk mengintegrasikan semua operation handlers."""
    
    @handle_ui_errors(error_component_title="Download Handler Manager Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize download handler manager.
        
        Args:
            ui_components: Dictionary UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
        
        # Initialize operation handlers
        self.download_handler = DownloadOperationHandler(ui_components=ui_components)
        self.check_handler = CheckOperationHandler(ui_components=ui_components)
        self.cleanup_handler = CleanupOperationHandler(ui_components=ui_components)
    
    @handle_ui_errors(error_component_title="Download Button Error", log_error=True)
    def handle_download_button(self, ui_config: Dict[str, Any]) -> None:
        """Handle download button click.
        
        Args:
            ui_config: Config dari UI yang sudah divalidasi
        """
        self.log_info("🔄 Memproses permintaan download dataset")
        
        # Check existing dataset
        existing_count = self._get_existing_dataset_count()
        
        # Show confirmation ONLY if existing dataset found
        if existing_count > 0:
            self._show_download_confirmation(ui_config, existing_count)
        else:
            # Execute download directly if no existing dataset
            self._execute_download_operation(ui_config)
    
    @handle_ui_errors(error_component_title="Check Button Error", log_error=True)
    def handle_check_button(self) -> None:
        """Handle check button click."""
        self.log_info("🔄 Memproses permintaan check dataset")
        self._execute_check_operation()
    
    @handle_ui_errors(error_component_title="Cleanup Button Error", log_error=True)
    def handle_cleanup_button(self) -> None:
        """Handle cleanup button click (dataset removal)."""
        self.log_info("🔄 Memproses permintaan cleanup dataset")
        
        # Get cleanup targets
        targets_result = self.cleanup_handler.get_cleanup_targets()
        
        if not self.is_success_response(targets_result):
            self.log_error("Gagal mendapatkan cleanup targets")
            return
        
        # Check if there are targets to cleanup
        if not targets_result.get('targets'):
            self.log_info("Tidak ada file untuk dibersihkan")
            return
        
        # ALWAYS show confirmation dialog for cleanup (even if no targets)
        self._show_cleanup_confirmation(targets_result.get('targets'))
    
    def _show_cleanup_confirmation(self, targets: List[str]) -> None:
        """Show cleanup confirmation dialog.
        
        Args:
            targets: List of files/directories to cleanup
        """
        target_count = len(targets) if targets else 0
        message = (
            f"Yakin ingin menghapus dataset? ({target_count:,} file/folder)\n\n"
            f"🗑️ Operasi ini akan menghapus semua file dataset\n"
            f"⚠️ Tindakan ini tidak dapat dibatalkan\n\n"
            "Lanjutkan cleanup?"
        )
        
        self.show_confirmation_dialog(
            self.ui_components,
            message=message,
            callback=lambda: self._execute_cleanup_operation(targets),
            title="Konfirmasi Cleanup Dataset",
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
    
    @handle_ui_errors(error_component_title="Download Operation Error", log_error=True)
    def _execute_download_operation(self, ui_config: Dict[str, Any]) -> None:
        """Execute download operation.
        
        Args:
            ui_config: Config dari UI yang sudah divalidasi
        """
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute download
        result = self.download_handler.execute_download(ui_config)
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    @handle_ui_errors(error_component_title="Check Operation Error", log_error=True)
    def _execute_check_operation(self) -> None:
        """Execute check operation."""
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute check
        result = self.check_handler.execute_check()
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def _execute_cleanup_operation(self, targets: List[str]) -> None:
        """Execute cleanup operation (dataset removal).
        
        Args:
            targets: List of files/directories to cleanup
        """
        # Disable buttons during operation
        self.disable_buttons()
        
        # Execute cleanup
        result = self.cleanup_handler.execute_cleanup({'targets': targets})
        
        # Enable buttons after operation
        self.enable_buttons()
        
        # Clear progress tracker
        self._reset_progress_tracker()
    
    @handle_ui_errors(error_component_title="Dataset Count Error", log_error=True)
    def _get_existing_dataset_count(self) -> int:
        """Get existing dataset count.
        
        Returns:
            Jumlah file dataset yang sudah ada
        """
        from smartcash.ui.dataset.downloader.services.backend_utils import get_existing_dataset_count
        return get_existing_dataset_count(self.logger)
    
    @handle_ui_errors(error_component_title="Confirmation Dialog Error", log_error=True)
    def _show_download_confirmation(self, ui_config: Dict[str, Any], existing_count: int) -> None:
        """Show download confirmation dialog.
        
        Args:
            ui_config: Config dari UI yang sudah divalidasi
            existing_count: Jumlah file dataset yang sudah ada
        """
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        message = (
            f"Dataset existing akan ditimpa! ({existing_count:,} gambar)\n\n"
            f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}\n"
            f"🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'}\n"
            f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}\n"
            f"💾 Backup: {'✅' if download.get('backup_existing', False) else '❌'}\n\n"
            "Lanjutkan download dataset?"
        )
        
        self.show_confirmation_dialog(
            self.ui_components,
            message=message,
            callback=lambda: self._execute_download_operation(ui_config),
            title="Konfirmasi Download Dataset",
            confirm_text="Ya, Download",
            cancel_text="Batal",
            danger_mode=True
        )
    
    @handle_ui_errors(error_component_title="Progress Tracker Error", log_error=True)
    def _reset_progress_tracker(self) -> None:
        """Reset progress tracker."""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    
    @handle_ui_errors(error_component_title="Setup Handlers Error", log_error=True)
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup handlers untuk integrasi dengan DownloaderInitializer.
        
        Returns:
            Dictionary berisi UI components yang sudah diupdate dengan handlers
        """
        from smartcash.ui.dataset.downloader.handlers.download_handler import (
            _handle_download_button,
            _handle_check_button,
            _handle_cleanup_button
        )
        
        # Setup button handlers
        download_button = self.ui_components.get('download_button')
        check_button = self.ui_components.get('check_button')
        cleanup_button = self.ui_components.get('cleanup_button')
        
        if download_button and hasattr(download_button, 'on_click'):
            download_button.on_click(lambda b: _handle_download_button(self.ui_components, self))
            
        if check_button and hasattr(check_button, 'on_click'):
            check_button.on_click(lambda b: _handle_check_button(self.ui_components, self))
            
        if cleanup_button and hasattr(cleanup_button, 'on_click'):
            cleanup_button.on_click(lambda b: _handle_cleanup_button(self.ui_components, self))
        
        # Store handler manager in UI components
        self.ui_components['handler_manager'] = self
        
        # Ensure summary container exists
        if 'summary_container' not in self.ui_components:
            from ipywidgets import HTML
            self.ui_components['summary_container'] = HTML(value="")
        
        return self.ui_components
