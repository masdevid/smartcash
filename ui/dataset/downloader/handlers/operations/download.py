"""
Download operation handler for dataset downloader.
"""
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.dataset.downloader.handlers.confirmation import confirmation_handler
from smartcash.ui.dataset.downloader.handlers.error_handling import handle_downloader_error, ErrorContext
from smartcash.ui.dataset.downloader.utils.ui_utils import log_to_accordion, clear_outputs
from smartcash.ui.dataset.downloader.utils.backend_utils import create_backend_downloader
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager


class DownloadOperation:
    """Handler for dataset download operations."""

    def __init__(self, ui_components: Dict[str, Any]):
        """Initialize with UI components."""
        self.ui_components = ui_components
        self._setup_operation_handlers()

    def setup_download_handler(self, config: Dict[str, Any]) -> None:
        """Set up the download button click handler."""
        download_button = self.ui_components.get('download_button')
        if download_button:
            download_button.on_click(lambda b: self._on_download_clicked(config))

    def _on_download_clicked(self, config: Dict[str, Any]) -> None:
        """Handle download button click."""
        print("\n=== _on_download_clicked: Starting download process")
        button_manager = get_button_manager(self.ui_components)
        clear_outputs(self.ui_components)
        button_manager.disable_buttons('download_button')
        
        try:
            print("=== _on_download_clicked: Getting config handler")
            config_handler = self.ui_components.get('config_handler')
            print(f"=== _on_download_clicked: Config handler: {config_handler is not None}")
            
            print("=== _on_download_clicked: Extracting config")
            ui_config = config_handler.extract_config(self.ui_components)
            print(f"=== _on_download_clicked: Extracted config: {bool(ui_config)}")
            
            print("=== _on_download_clicked: Validating config")
            validation = config_handler.validate_config(ui_config)
            print(f"=== _on_download_clicked: Config valid: {validation['valid']}")
            
            if not validation['valid']:
                error_msg = f"Config tidak valid: {', '.join(validation['errors'])}"
                log_to_accordion(self.ui_components, f"❌ {error_msg}", 'error')
                print(f"=== _on_download_clicked: Config validation failed: {error_msg}")
                return
                
            # Check for existing files if needed
            print("=== _on_download_clicked: Checking for existing files")
            existing_count = self._check_existing_files(ui_config)
            print(f"=== _on_download_clicked: Existing files count: {existing_count}")
            
            if existing_count is not None:
                print("=== _on_download_clicked: Showing download confirmation")
                self._show_download_confirmation(ui_config, existing_count)
            
        except Exception as e:
            self._handle_download_error(e, "download_dataset")
            button_manager.enable_buttons()

    def _check_existing_files(self, ui_config: Dict[str, Any]) -> Optional[int]:
        """Check for existing files in target directory."""
        # Implementation depends on your specific file checking logic
        # Return number of existing files or None if check is not needed
        return 0  # Placeholder

    def _show_download_confirmation(self, ui_config: Dict[str, Any], existing_count: int) -> None:
        """Show download confirmation dialog."""
        print("\n=== _show_download_confirmation: Starting")
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        message = """
        <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
            <h3 style='margin-top: 0; color: #2c3e50;'>Download Dataset</h3>
            <div style='background: #f8f9fa; padding: 12px; border-radius: 6px; margin: 12px 0; font-size: 14px;'>
                <div style='margin-bottom: 8px;'><strong>Dataset:</strong> {dataset_name} v{version}</div>
                <div style='margin-bottom: 8px;'><strong>Format:</strong> {format} ({split_type} split)</div>
                <div><strong>Lokasi:</strong> {download_dir}</div>
            </div>
            {existing_warning}
            <p style='margin: 12px 0 0; color: #6c757d;'>Apakah Anda yakin ingin melanjutkan download?</p>
        </div>
        """.format(
            dataset_name=roboflow.get('dataset_name', 'Tidak Diketahui'),
            version=roboflow.get('version', '1'),
            format=download.get('format', 'yolov8'),
            split_type=download.get('split_type', 'train'),
            download_dir=download.get('download_dir', 'tidak diketahui'),
            existing_warning=f"<div style='background: #fff3cd; color: #856404; padding: 8px; border-radius: 4px; margin: 8px 0 0; font-size: 13px;'>⚠️ {existing_count} file sudah ada</div>" if existing_count > 0 else ""
        )
        
        print("=== _show_download_confirmation: Calling confirmation_handler.show_confirmation_dialog")
        try:
            confirmation_handler.show_confirmation_dialog(
                ui_components=self.ui_components,
                title="Konfirmasi Download",
                message=message,
                confirm_callback=self._execute_download,
                confirm_args=(ui_config,),
                danger_mode=False
            )
            print("=== _show_download_confirmation: confirmation_dialog called successfully")
        except Exception as e:
            print(f"=== _show_download_confirmation: Error showing confirmation dialog: {str(e)}")
            raise

    def _execute_download(self, ui_config: Dict[str, Any]) -> None:
        """Execute the download operation."""
        try:
            # Setup operation environment
            self._setup_operation("download")
            log_to_accordion(self.ui_components, "✅ Download dikonfirmasi, memulai...", "success")
            
            # Start operation in background thread
            with ThreadPoolExecutor() as executor:
                executor.submit(self._perform_download, ui_config)
                
        except Exception as e:
            self._handle_download_error(e, "execute_download")
            button_manager = get_button_manager(self.ui_components)
            if button_manager:
                button_manager.enable_buttons()

    def _perform_download(self, ui_config: Dict[str, Any]) -> None:
        """Perform the actual download operation."""
        button_manager = get_button_manager(self.ui_components)
        try:
            # Create downloader instance
            downloader = create_backend_downloader(
                ui_config,
                self.ui_components.get('logger_bridge')
            )
            
            if not downloader:
                raise ValueError("Gagal membuat downloader instance")
            
            # Execute download
            result = downloader.download()
            
            if result.get('status') is True:
                log_to_accordion(self.ui_components, "✅ Download berhasil diselesaikan", 'success')
            else:
                error_msg = result.get('error', 'Gagal mendownload dataset')
                log_to_accordion(self.ui_components, f"❌ {error_msg}", 'error')
                
        except Exception as e:
            self._handle_download_error(e, "perform_download")
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

    def _handle_download_error(self, error: Exception, operation: str) -> None:
        """Handle download-related errors."""
        from smartcash.ui.dataset.downloader.handlers.error_handling import (
            handle_downloader_error, 
            create_error_context
        )
        
        handle_downloader_error(
            error,
            create_error_context(
                operation=operation,
                component="downloader",
                ui_components=self.ui_components
            ),
            logger=self.ui_components.get('logger_bridge'),
            ui_components=self.ui_components
        )

    def _setup_operation_handlers(self) -> None:
        """Setup operation-specific handlers."""
        # Register operation handlers
        self.ui_components['_operation_handlers'] = {
            'download': self._execute_download,
        }
