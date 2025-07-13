"""
File: smartcash/ui/dataset/downloader/handlers/operation/cleanup.py
Deskripsi: Handler untuk operasi cleanup dataset dengan centralized error handling
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.dataset.downloader.operations.base_operation import BaseDownloaderHandler
from smartcash.ui.core.decorators import handle_ui_errors

class CleanupOperationHandler(BaseDownloaderHandler):
    """Handler untuk operasi cleanup dataset dengan centralized error handling."""
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize cleanup operation handler.
        
        Args:
            ui_components: Dictionary UI components
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(ui_components=ui_components, **kwargs)
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def get_cleanup_targets(self) -> Dict[str, Any]:
        """Get cleanup targets dari backend service.
        
        Returns:
            Dictionary dengan status dan cleanup targets
        """
        self.logger.info("🔍 Mencari file yang dapat dibersihkan")
        
        from smartcash.ui.dataset.downloader.services import get_dataset_scanner
        scanner = get_dataset_scanner()
        targets_result = scanner.get_cleanup_targets()
        
        if not targets_result or 'summary' not in targets_result:
            self.logger.error("Gagal mendapatkan cleanup targets - respons tidak valid")
            return {'status': False, 'message': "Gagal mendapatkan cleanup targets - respons tidak valid"}
        
        summary = targets_result.get('summary', {})
        total_files = summary.get('total_files', 0)
        
        if total_files == 0:
            self.logger.info("Tidak ada file untuk dibersihkan")
            return {'status': True, 'message': "Tidak ada file untuk dibersihkan", 'targets': None}
        
        self.logger.info(f"Ditemukan {total_files} file yang dapat dibersihkan")
        return {'status': True, 'message': f"Ditemukan {total_files} file yang dapat dibersihkan", 'targets': targets_result}
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error", log_error=True)
    def execute_cleanup(self, targets_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cleanup operation dengan backend service.
        
        Args:
            targets_result: Hasil dari get_cleanup_targets
            
        Returns:
            Dictionary dengan status operasi
        """
        self.logger.info("🧹 Memulai pembersihan dataset")
        
        # Setup progress tracker
        self._setup_progress_tracker("Dataset Cleanup")
        
        # Create cleanup service
        cleanup_service = self._create_cleanup_service()
        if not cleanup_service:
            self.logger.error("Gagal membuat cleanup service")
            return {'status': False, 'message': "Gagal membuat cleanup service"}
        
        # Setup progress callback
        if hasattr(cleanup_service, 'set_progress_callback') and 'progress_callback' in self.ui_components:
            cleanup_service.set_progress_callback(self.ui_components['progress_callback'])
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset()
        
        if result and result.get('status') == 'success':
            self._update_summary_container(result, targets_result)
            self.logger.info("✅ Pembersihan dataset selesai")
            return {
                'success': True, 
                'message': "Pembersihan dataset selesai", 
                'deleted_files': result.get('deleted_files', 0),
                'freed_space': result.get('freed_space', '0B')
            }
        else:
            error_msg = result.get('message', 'Pembersihan gagal') if result else 'No response from service'
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False, 
                'message': error_msg, 
                'deleted_files': 0,
                'freed_space': '0B'
            }
    
    @handle_ui_errors(error_component_title="Confirmation Dialog Error", log_error=True)
    def show_cleanup_confirmation(self, targets_result: Dict[str, Any], on_confirm: Callable) -> None:
        """Show cleanup confirmation dialog.
        
        Args:
            targets_result: Hasil dari get_cleanup_targets
            on_confirm: Callback function ketika user konfirmasi
        """
        summary = targets_result.get('summary', {})
        total_files = summary.get('total_files', 0)
        total_size = summary.get('total_size_mb', 0)
        
        message = (
            f"Akan menghapus {total_files} file ({total_size:.2f} MB).\n\n"
            "Operasi ini tidak dapat dibatalkan. Lanjutkan?"
        )
        
        self.show_confirmation_dialog(
            self.ui_components,
            message=message,
            callback=on_confirm,
            title="Konfirmasi Pembersihan Dataset",
            confirm_text="Bersihkan",
            cancel_text="Batal",
            danger_mode=True
        )
    
    @handle_ui_errors(error_component_title="Progress Tracker Error", log_error=True)
    def _setup_progress_tracker(self, operation_name: str):
        """Setup progress tracker untuk operation."""
        progress_tracker = self.ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    @handle_ui_errors(error_component_title="Backend Service Error", log_error=True)
    def _create_cleanup_service(self):
        """Create cleanup service."""
        from smartcash.ui.dataset.downloader.services import get_dataset_scanner
        scanner = get_dataset_scanner()
        cleanup_service = scanner.create_cleanup_service()
        return cleanup_service
        
    @handle_ui_errors(error_component_title="Summary Update Error", log_error=True)
    def _update_summary_container(self, result: Dict[str, Any], targets_result: Dict[str, Any]):
        """Update summary container dengan hasil cleanup.
        
        Args:
            result: Hasil dari operasi cleanup
            targets_result: Hasil dari get_cleanup_targets
        """
        summary_container = self.ui_components.get('summary_container')
        if not summary_container:
            self.log_debug("Summary container tidak tersedia")
            return
            
        # Extract summary data
        summary = targets_result.get('summary', {})
        total_files = summary.get('total_files', 0)
        total_size_mb = summary.get('total_size_mb', 0)
        targets = targets_result.get('targets', {})
        
        # Format summary content
        summary_lines = [
            "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
            "<h3>📊 Ringkasan Pembersihan Dataset</h3>",
            f"<p>🗑️ Total file dihapus: <b>{total_files:,}</b></p>",
            f"<p>💾 Total ukuran dibebaskan: <b>{total_size_mb:.2f} MB</b></p>"
        ]
        
        # Add target details if available
        if targets:
            summary_lines.append("<p>🔍 Detail pembersihan:</p>")
            summary_lines.append("<ul>")
            for target_name, target_info in targets.items():
                file_count = target_info.get('file_count', 0)
                size_formatted = target_info.get('size_formatted', '0 B')
                summary_lines.append(f"<li>{target_name}: {file_count:,} file ({size_formatted})</li>")
            summary_lines.append("</ul>")
        
        summary_lines.append("</div>")
        
        # Update summary container
        summary_container.clear_output()
        summary_container.append_html("".join(summary_lines))
