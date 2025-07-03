"""
File: smartcash/ui/dataset/downloader/handlers/operation/download.py
Deskripsi: Handler untuk operasi download dataset dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.handlers.base_downloader_handler import BaseDownloaderHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors

class DownloadOperationHandler(BaseDownloaderHandler):
    """Handler untuk operasi download dataset dengan centralized error handling."""
    
    @handle_ui_errors(error_component_title="Download Operation Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize download operation handler.
        
        Args:
            ui_components: Dictionary UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
    
    @handle_ui_errors(error_component_title="Download Operation Error", log_error=True)
    def execute_download(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download operation dengan backend service.
        
        Args:
            ui_config: Config dari UI yang sudah divalidasi
            
        Returns:
            Dictionary dengan status operasi
        """
        self.log_info("🚀 Memulai download dataset")
        
        # Setup progress tracker
        self._setup_progress_tracker("Dataset Download")
        
        # Create downloader
        downloader = self._create_backend_downloader(ui_config)
        if not downloader:
            self.log_error("Gagal membuat download service")
            return {'status': False, 'message': "Gagal membuat download service"}
        
        # Setup progress callback
        if hasattr(downloader, 'set_progress_callback') and 'progress_callback' in self.ui_components:
            downloader.set_progress_callback(self.ui_components['progress_callback'])
        
        # Log config
        self._log_download_config(ui_config)
        
        # Execute download
        result = downloader.download_dataset()
        
        if result and result.get('status') == 'success':
            self._show_download_success(result)
            self._update_summary_container(result)
            self.log_info("✅ Download berhasil")
            return {'status': True, 'message': "Download berhasil", 'result': result}
        else:
            error_msg = result.get('message', 'Download gagal') if result else 'No response from service'
            self.log_error(f"❌ {error_msg}")
            return {'status': False, 'message': error_msg, 'result': result}
    
    @handle_ui_errors(error_component_title="Progress Tracker Error", log_error=True)
    def _setup_progress_tracker(self, operation_name: str):
        """Setup progress tracker untuk operation."""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    @handle_ui_errors(error_component_title="Backend Service Error", log_error=True)
    def _create_backend_downloader(self, ui_config: Dict[str, Any]):
        """Create backend downloader service."""
        from smartcash.ui.dataset.downloader.utils.backend_utils import create_backend_downloader
        return create_backend_downloader(ui_config, self.logger)
    
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def _log_download_config(self, ui_config: Dict[str, Any]):
        """Log download config ke UI menggunakan parent class logging methods."""
        # Log config details using parent class logging methods
        self.log_info("📋 Konfigurasi Download:")
        
        # Log source type
        source_type = ui_config.get('source_type', 'unknown')
        self.log_info(f"🔹 Tipe sumber: {source_type}")
        
        # Log dataset path
        dataset_path = ui_config.get('dataset_path', '')
        self.log_info(f"🔹 Path dataset: {dataset_path}")
        
        # Log additional options
        options = ui_config.get('options', {})
        if options:
            self.log_info("🔹 Opsi tambahan:")
            for key, value in options.items():
                if isinstance(value, bool) and value:
                    self.log_info(f"  - {key}: {value}")
    
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def _show_download_success(self, result: Dict[str, Any]):
        """Show download success di UI menggunakan parent class methods."""
        # Get summary info from result
        summary = result.get('summary', {})
        dataset_path = result.get('dataset_path', '')
        
        # Show success message using parent class methods
        success_message = f"✅ Download berhasil ke {dataset_path}"
        
        # Update status panel with success message
        self.update_status_panel(
            self.ui_components,
            message=success_message,
            status_type="success",
            title="Download Berhasil"
        )
        
        # Show success info dialog
        total_images = summary.get('total_images', 0)
        total_labels = summary.get('total_labels', 0)
        info_message = f"Dataset berhasil didownload ke {dataset_path}\n\n" \
                      f"Total gambar: {total_images}\n" \
                      f"Total label: {total_labels}"
        
        self.show_info_dialog(
            self.ui_components,
            message=info_message,
            title="Download Berhasil",
            ok_text="OK"
        )
        
    @handle_ui_errors(error_component_title="Summary Update Error", log_error=True)
    def _update_summary_container(self, result: Dict[str, Any]):
        """Update summary container dengan hasil download.
        
        Args:
            result: Hasil dari operasi download
        """
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.log_debug("Summary container tidak tersedia")
            return
            
        # Extract summary data
        stats = result.get('stats', {})
        total_images = stats.get('total_images', 0)
        total_annotations = stats.get('total_annotations', 0)
        
        # Format summary content
        summary_lines = [
            "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
            "<h3>📊 Ringkasan Download Dataset</h3>",
            f"<p>✅ Total gambar: <b>{total_images:,}</b></p>",
            f"<p>🏷️ Total anotasi: <b>{total_annotations:,}</b></p>"
        ]
        
        # Add class distribution if available
        class_distribution = stats.get('class_distribution', {})
        if class_distribution:
            summary_lines.append("<p>🔍 Distribusi kelas:</p>")
            summary_lines.append("<ul>")
            for class_name, count in class_distribution.items():
                summary_lines.append(f"<li>{class_name}: {count:,}</li>")
            summary_lines.append("</ul>")
        
        summary_lines.append("</div>")
        
        # Update summary container
        summary_container.clear_output()
        summary_container.append_html("".join(summary_lines))
