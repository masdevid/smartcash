"""
Cleanup operation handler for dataset downloader.
"""
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.dataset.downloader.handlers.confirmation import confirmation_handler
from smartcash.ui.dataset.downloader.utils.ui_utils import log_to_accordion
from smartcash.ui.dataset.downloader.utils.backend_utils import get_cleanup_targets, create_backend_cleanup_service


class CleanupOperation:
    """Handler for dataset cleanup operations."""

    def __init__(self, ui_components: Dict[str, Any]):
        """Initialize with UI components."""
        self.ui_components = ui_components
        self._setup_operation_handlers()

    def setup_cleanup_handler(self, config: Dict[str, Any]) -> None:
        """Set up the cleanup button click handler."""
        cleanup_button = self.ui_components.get('cleanup_button')
        if cleanup_button:
            cleanup_button.on_click(lambda b: self._on_cleanup_clicked())

    def _on_cleanup_clicked(self) -> None:
        """Handle cleanup button click."""
        button_manager = get_button_manager(self.ui_components)
        clear_outputs(self.ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            # Get cleanup targets
            targets_result = get_cleanup_targets(self.ui_components.get('logger_bridge'))
            
            if targets_result.get('has_targets', False):
                self._show_cleanup_confirmation(targets_result)
            else:
                log_to_accordion(self.ui_components, "ℹ️ Tidak ada target cleanup yang ditemukan", 'info')
                
        except Exception as e:
            self._handle_cleanup_error(e, "check_cleanup_targets")
            button_manager.enable_buttons()

    def _show_cleanup_confirmation(self, targets_result: Dict[str, Any]) -> None:
        """Show cleanup confirmation dialog."""
        files = targets_result.get('files', [])
        file_count = len(files)
        
        # Build confirmation message
        message = """
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
            <h3 style='margin: 0 0 12px 0; color: #2c3e50;'>Hapus File Dataset</h3>
            <div style='background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                <div style='margin-bottom: 8px;'>{file_count} file akan dihapus:</div>
                <div style='background: white; padding: 8px; border-radius: 4px; max-height: 150px; overflow-y: auto; font-size: 13px;'>
                    {file_list}
                </div>
                {more_files}
            </div>
            <div style='background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; margin-bottom: 12px; font-size: 13px;'>
                ⚠️ Aksi ini tidak dapat dibatalkan
            </div>
            <p style='margin: 0;'>Lanjutkan penghapusan?</p>
        </div>
        """.format(
            file_count=file_count,
            file_list='\n'.join([
                f"<div style='padding: 4px 0;'>{file}</div>" 
                for file in files[:10]]
            ),
            more_files=f"<div style='color: #6c757d; padding: 4px 0;'>... dan {file_count - 10} file lainnya</div>" 
                     if file_count > 10 else ""
        )

        confirmation_handler.show_confirmation_dialog(
            ui_components=self.ui_components,
            title="Konfirmasi Pembersihan",
            message=message,
            confirm_callback=self._execute_cleanup,
            confirm_args=(targets_result,),
            danger_mode=True  # Use danger mode for destructive actions
        )

    def _execute_cleanup(self, targets_result: Dict[str, Any]) -> None:
        """Execute the cleanup operation."""
        try:
            # Setup operation environment
            self._setup_operation("cleanup")
            log_to_accordion(self.ui_components, "✅ Cleanup dikonfirmasi, memulai...", "success")
            
            # Start operation in background thread
            with ThreadPoolExecutor() as executor:
                executor.submit(self._perform_cleanup, targets_result)
                
        except Exception as e:
            self._handle_cleanup_error(e, "execute_cleanup")
            button_manager = get_button_manager(self.ui_components)
            if button_manager:
                button_manager.enable_buttons()

    def _perform_cleanup(self, targets_result: Dict[str, Any]) -> None:
        """Perform the actual cleanup operation."""
        button_manager = get_button_manager(self.ui_components)
        try:
            # Create cleanup service
            cleanup_service = create_backend_cleanup_service(
                self.ui_components.get('logger_bridge')
            )
            
            if not cleanup_service:
                raise ValueError("Gagal membuat cleanup service")
            
            # Execute cleanup
            result = cleanup_service.cleanup(targets_result.get('files', []))
            
            if result.get('status') is True:
                log_to_accordion(self.ui_components, "✅ Pembersihan berhasil diselesaikan", 'success')
            else:
                error_msg = result.get('error', 'Gagal membersihkan file')
                log_to_accordion(self.ui_components, f"❌ {error_msg}", 'error')
                
        except Exception as e:
            self._handle_cleanup_error(e, "perform_cleanup")
        finally:
            if button_manager:
                button_manager.enable_buttons()

    def _setup_operation(self, operation_name: str) -> None:
        """Setup operation environment."""
        # Clear log output
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output()
        
        # Setup progress tracker
        self._setup_progress_tracker(operation_name)
        
        # Log operation start
        log_to_accordion(self.ui_components, f"🚀 Memulai {operation_name.lower()}...", 'info')

    def _setup_progress_tracker(self, operation_name: str) -> None:
        """Setup and initialize progress tracker."""
        if 'progress_tracker' not in self.ui_components:
            from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
            from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
            
            config = ProgressConfig()
            config.add_level(ProgressLevel("overall", f"{operation_name.capitalize()} Dataset"))
            self.ui_components['progress_tracker'] = ProgressTracker(config=config)
        
        progress_tracker = self.ui_components['progress_tracker']
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
        
        if hasattr(progress_tracker, 'container'):
            progress_tracker.container.layout.display = 'block'

    def _handle_cleanup_error(self, error: Exception, operation: str) -> None:
        """Handle cleanup-related errors."""
        from .error_handling import handle_downloader_error, create_downloader_error_context
        
        handle_downloader_error(
            error,
            create_downloader_error_context(
                operation=f"{operation}_cleanup",
                ui_components=self.ui_components
            ),
            logger=self.ui_components.get('logger_bridge'),
            ui_components=self.ui_components
        )

    def _setup_operation_handlers(self) -> None:
        """Setup operation-specific handlers."""
        # Register operation handlers
        if '_operation_handlers' not in self.ui_components:
            self.ui_components['_operation_handlers'] = {}
            
        self.ui_components['_operation_handlers'].update({
            'cleanup': self._execute_cleanup,
        })
