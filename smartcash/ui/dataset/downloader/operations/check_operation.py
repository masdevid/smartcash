"""
File: smartcash/ui/dataset/downloader/handlers/operation/check.py
Deskripsi: Handler untuk operasi check dataset dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.handlers.base_downloader_handler import BaseDownloaderHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors

class CheckOperationHandler(BaseDownloaderHandler):
    """Handler untuk operasi check dataset dengan centralized error handling."""
    
    @handle_ui_errors(error_component_title="Check Operation Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize check operation handler.
        
        Args:
            ui_components: Dictionary UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
    
    @handle_ui_errors(error_component_title="Check Operation Error", log_error=True)
    def execute_check(self) -> Dict[str, Any]:
        """Execute check operation dengan backend scanner.
        
        Returns:
            Dictionary dengan status operasi
        """
        self.logger.info("🔍 Memulai pemeriksaan dataset")
        
        # Setup progress tracker
        self._setup_progress_tracker("Dataset Check")
        
        # Create dataset scanner
        scanner = self._create_dataset_scanner()
        if not scanner:
            self.logger.error("Gagal membuat dataset scanner")
            return {
                'exists': False,
                'file_count': 0,
                'total_size': '0B',
                'status': False,
                'message': 'Gagal membuat dataset scanner'
            }
        
        # Setup progress callback if available
        if hasattr(scanner, 'set_progress_callback') and 'progress_callback' in self.ui_components:
            scanner.set_progress_callback(self.ui_components['progress_callback'])
        
        # Execute scan
        result = scanner.scan_existing_dataset_parallel()
        
        if result and result.get('status') == 'success':
            self._display_check_results(result)
            self._update_summary_container(result)
            self.logger.info("✅ Pemeriksaan dataset selesai")
            
            # Return format expected by the test
            return {
                'exists': True,
                'file_count': result.get('file_count', 0),
                'total_size': result.get('total_size', '0B'),
                'status': True,
                'message': 'Pemeriksaan dataset selesai'
            }
        else:
            error_msg = result.get('message', 'Pemeriksaan gagal') if result else 'Tidak ada respons dari scanner'
            self.logger.error(f"❌ {error_msg}")
            return {
                'exists': False,
                'file_count': 0,
                'total_size': '0B',
                'status': False,
                'message': error_msg
            }
    
    def _get_download_service(self):
        """Get download service from parent or create a new one."""
        # Try to get download service from parent
        if hasattr(self, 'parent') and hasattr(self.parent, 'get_download_service'):
            return self.parent.get_download_service()
            
        # Try to get from ui_components
        if 'download_service' in self.ui_components:
            return self.ui_components['download_service']
            
        # Create a new one as fallback
        try:
            from smartcash.ui.dataset.downloader.services.downloader_service import DownloaderService
            return DownloaderService()
        except Exception as e:
            self.logger.error(f"Gagal membuat download service: {e}")
            return None
    
    @handle_ui_errors(error_component_title="Progress Tracker Error", log_error=True)
    def _setup_progress_tracker(self, operation_name: str):
        """Setup progress tracker untuk operation."""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    @handle_ui_errors(error_component_title="Backend Service Error", log_error=True)
    def _create_dataset_scanner(self):
        """Create dataset scanner service."""
        from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
        return create_dataset_scanner(self.logger)
    
    @handle_ui_errors(error_component_title="UI Operation Error", log_error=True)
    def _display_check_results(self, result: Dict[str, Any]):
        """Display check results di UI."""
        from smartcash.ui.dataset.downloader.handlers.utils import display_check_results
        display_check_results(self.ui_components, result)
        
    @handle_ui_errors(error_component_title="Summary Update Error", log_error=True)
    def _update_summary_container(self, result: Dict[str, Any]):
        """Update summary container dengan hasil check.
        
        Args:
            result: Hasil dari operasi check
        """
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.logger.debug("Summary container tidak tersedia")
            return
            
        # Extract summary data
        stats = result.get('stats', {})
        total_images = stats.get('total_images', 0)
        total_annotations = stats.get('total_annotations', 0)
        issues = result.get('issues', [])
        
        # Format summary content
        summary_lines = [
            "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
            "<h3>📊 Ringkasan Pemeriksaan Dataset</h3>",
            f"<p>🖼️ Total gambar: <b>{total_images:,}</b></p>",
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
        
        # Add issues if any
        if issues:
            issue_count = len(issues)
            summary_lines.append(f"<p>⚠️ <b>{issue_count}</b> masalah terdeteksi:</p>")
            summary_lines.append("<ul>")
            for issue in issues[:5]:  # Show only top 5 issues in summary
                issue_type = issue.get('type', 'Unknown')
                issue_count = issue.get('count', 1)
                summary_lines.append(f"<li>{issue_type}: {issue_count} item</li>")
            
            if len(issues) > 5:
                summary_lines.append(f"<li>... dan {len(issues) - 5} masalah lainnya</li>")
            
            summary_lines.append("</ul>")
        else:
            summary_lines.append("<p>✅ Tidak ada masalah terdeteksi</p>")
        
        summary_lines.append("</div>")
        
        # Update summary container
        summary_container.clear_output()
        summary_container.append_html("".join(summary_lines))
