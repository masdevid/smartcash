"""
File: smartcash/ui/dataset/downloader/handlers/operation/download.py
Deskripsi: Handler untuk operasi download dataset dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.operations.base_operation import BaseDownloaderHandler
from smartcash.ui.core.decorators import handle_ui_errors

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
        self.logger.info("🚀 Memulai download dataset")
        
        # Setup progress tracker
        self._setup_progress_tracker("Dataset Download")
        
        # Create downloader
        downloader = self._create_backend_downloader(ui_config)
        if not downloader:
            self.logger.error("Gagal membuat download service")
            return {'status': False, 'message': "Gagal membuat download service"}
        
        # Setup progress callback
        if hasattr(downloader, 'set_progress_callback') and 'progress_callback' in self.ui_components:
            downloader.set_progress_callback(self.ui_components['progress_callback'])
        
        # Log config
        self._log_download_config(ui_config)
        
        # Execute download
        result = downloader.download_dataset()
        
        # Get progress tracker
        progress_tracker = self.ui_components.get('progress_tracker')
        
        try:
            if result and result.get('status') == 'success':
                # Complete the current stage
                if progress_tracker:
                    progress_tracker.complete_stage()
                    progress_tracker.update_overall(100, "✅ Download selesai")
                
                self._show_download_success(result)
                self._update_summary_container(result)
                download_path = result.get('dataset_path', '')
                self.logger.info(f"✅ Download berhasil ke {download_path}")
                return {'status': True, 'message': "Download berhasil", 'result': result}
            else:
                error_msg = result.get('message', 'Download gagal') if result else 'Tidak ada respons dari layanan'
                if progress_tracker:
                    progress_tracker.update_overall(0, f"❌ {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                return {'status': False, 'message': error_msg, 'result': result}
                
        except Exception as e:
            error_msg = f"Error saat proses download: {str(e)}"
            if progress_tracker:
                progress_tracker.update_overall(0, f"❌ {error_msg}")
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            return {'status': False, 'message': error_msg, 'result': result}
    
    @handle_ui_errors(error_component_title="Progress Tracker Error", log_error=True)
    def _setup_progress_tracker(self, operation_name: str):
        """Setup progress tracker untuk operation.
        
        Args:
            operation_name: Nama operasi yang akan ditampilkan di progress tracker
        """
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            # Start a new stage for this operation
            progress_tracker.start_stage(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
            
            # If there's a progress bar in UI components, update it
            if 'progress_bar' in self.ui_components and self.ui_components['progress_bar']:
                self.ui_components['progress_bar'].value = 0
                self.ui_components['progress_bar'].description = f"Memulai {operation_name.lower()}..."
    
    @handle_ui_errors(error_component_title="Backend Service Error", log_error=True)
    def _create_backend_downloader(self, ui_config: Dict[str, Any]):
        """Create backend downloader service."""
        from smartcash.ui.dataset.downloader.services import get_dataset_scanner
        scanner = get_dataset_scanner()
        return scanner.create_downloader(ui_config)
    
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def _log_download_config(self, ui_config: Dict[str, Any]):
        """Log download config ke UI menggunakan parent class logging methods."""
        # Log config details using parent class logging methods
        self.logger.info("📋 Konfigurasi Download:")
        
        # Log source type
        source_type = ui_config.get('source_type', 'unknown')
        self.logger.info(f"🔹 Tipe sumber: {source_type}")
        
        # Log dataset path
        dataset_path = ui_config.get('dataset_path', '')
        self.logger.info(f"🔹 Path dataset: {dataset_path}")
        
        # Log additional options
        options = ui_config.get('options', {})
        if options:
            self.logger.info("🔹 Opsi tambahan:")
            for key, value in options.items():
                if isinstance(value, bool) and value:
                    self.logger.info(f"  - {key}: {value}")
    
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def _show_download_success(self, result: Dict[str, Any]):
        """Show download success di UI menggunakan parent class methods."""
        # Get summary info from result
        summary = result.get('summary', {})
        dataset_path = result.get('dataset_path', '')
        
        # Log success message
        success_message = f"✅ Download berhasil ke {dataset_path}"
        self.logger.info(success_message)
        
        # Update status panel if available in UI components
        if 'status_panel' in self.ui_components and self.ui_components['status_panel']:
            self.ui_components['status_panel'].update(
                message=success_message,
                status_type="success",
                title="Download Berhasil"
            )
        
        # Log detailed summary
        total_images = summary.get('total_images', 0)
        total_labels = summary.get('total_labels', 0)
        
        # Log detailed information
        self.logger.info(f"📊 Download Summary:")
        self.logger.info(f"  - Dataset Path: {dataset_path}")
        self.logger.info(f"  - Total Images: {total_images}")
        self.logger.info(f"  - Total Labels: {total_labels}")
        
        # If we have a summary container, update it
        if 'summary_container' in self.ui_components and self.ui_components['summary_container']:
            self._update_summary_container(result)
        
    @handle_ui_errors(error_component_title="Summary Update Error", log_error=True)
    def _update_summary_container(self, result: Dict[str, Any]):
        """Update summary container dengan hasil download.
        
        Args:
            result: Hasil dari operasi download
        """
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.logger.debug("Summary container tidak tersedia")
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
