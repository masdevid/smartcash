"""
File: smartcash/ui/dataset/download/services/download_execution_service.py
Deskripsi: Service untuk execution download dengan confirmation handling dan progress integration
"""

from typing import Dict, Any, Callable
from IPython.display import display
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from smartcash.ui.dataset.download.services.ui_download_service import UIDownloadService
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status

class DownloadExecutionService:
    """Service untuk execution download dengan enhanced confirmation dan progress."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.download_service = UIDownloadService(ui_components)
    
    def execute_with_confirmation(self, params: Dict[str, Any], completion_callback: Callable = None) -> None:
        """Execute download dengan comprehensive confirmation handling."""
        try:
            # Check existing dataset
            existing_check = check_complete_dataset_status()
            
            if self._has_existing_dataset(existing_check):
                self._show_replacement_confirmation(params, existing_check, completion_callback)
            else:
                self._show_standard_confirmation(params, completion_callback)
                
        except Exception as e:
            self.logger and self.logger.error(f"❌ Error execution: {str(e)}")
            if completion_callback:
                completion_callback(self.ui_components, {'status': 'error', 'message': str(e)})
    
    def execute_direct(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Direct execution tanpa confirmation untuk programmatic use."""
        try:
            self.logger and self.logger.info("🚀 Direct execution download")
            return self.download_service.download_dataset(params)
        except Exception as e:
            error_msg = f"Direct execution error: {str(e)}"
            self.logger and self.logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'message': error_msg}
    
    def _has_existing_dataset(self, dataset_status: Dict[str, Any]) -> bool:
        """Check apakah ada existing dataset yang significant."""
        final_dataset = dataset_status.get('final_dataset', {})
        return final_dataset.get('exists', False) and final_dataset.get('total_images', 0) > 0
    
    def _show_replacement_confirmation(self, params: Dict[str, Any], existing_info: Dict[str, Any], 
                                     completion_callback: Callable = None) -> None:
        """Show confirmation untuk replace existing dataset."""
        final_dataset = existing_info.get('final_dataset', {})
        storage_info = existing_info.get('storage_info', {})
        
        # Build split info
        split_info_lines = []
        for split, stats in final_dataset.get('splits', {}).items():
            if stats.get('exists', False) and stats.get('images', 0) > 0:
                split_info_lines.append(f"• {split}: {stats['images']} gambar, {stats['labels']} label")
        
        message = (
            f"⚠️ Dataset sudah ada di lokasi target!\n\n"
            f"📊 Dataset yang ada:\n" + '\n'.join(split_info_lines) + 
            f"\n• Total: {final_dataset.get('total_images', 0)} gambar\n"
            f"• Storage: {storage_info.get('type', 'Unknown')}\n\n"
            f"📥 Dataset baru:\n"
            f"• Workspace: {params['workspace']}\n"
            f"• Project: {params['project']}\n"
            f"• Version: {params['version']}\n\n"
            f"🔄 Dataset yang ada akan diganti dengan yang baru.\n"
            f"Lanjutkan download?"
        )
        
        def on_confirm(b):
            self._clear_confirmation_area()
            self._execute_confirmed_download(params, completion_callback)
        
        def on_cancel(b):
            self._clear_confirmation_area()
            self.logger and self.logger.info("❌ Download dibatalkan oleh user")
            if 'error_operation' in self.ui_components:
                self.ui_components['error_operation']("Download dibatalkan")
        
        dialog = create_confirmation_dialog(
            title="⚠️ Konfirmasi Replace Dataset",
            message=message, on_confirm=on_confirm, on_cancel=on_cancel,
            confirm_text="Ya, Replace Dataset", cancel_text="Batal", danger_mode=True
        )
        
        self._show_confirmation_dialog(dialog)
    
    def _show_standard_confirmation(self, params: Dict[str, Any], completion_callback: Callable = None) -> None:
        """Show standard download confirmation."""
        env_manager = self.ui_components.get('env_manager')
        
        if env_manager and env_manager.is_drive_mounted:
            storage_info = f"📁 Storage: Google Drive ({env_manager.drive_path})"
            storage_note = "💾 Dataset akan tersimpan permanen di Google Drive"
        else:
            storage_info = "📁 Storage: Local Storage"
            storage_note = "⚠️ Dataset akan hilang saat runtime restart"
        
        message = (
            f"📥 Konfirmasi Download Dataset\n\n"
            f"🎯 Dataset yang akan didownload:\n"
            f"• Workspace: {params['workspace']}\n"
            f"• Project: {params['project']}\n"
            f"• Version: {params['version']}\n"
            f"• Output: {params['output_dir']}\n\n"
            f"💾 Storage Info:\n"
            f"• {storage_info}\n"
            f"• {storage_note}\n\n"
            f"🚀 Proses akan:\n"
            f"1. Download dataset dari Roboflow\n"
            f"2. Ekstrak dan organisir ke struktur final\n"
            f"3. Validasi hasil download\n\n"
            f"Lanjutkan download?"
        )
        
        def on_confirm(b):
            self._clear_confirmation_area()
            self._execute_confirmed_download(params, completion_callback)
        
        def on_cancel(b):
            self._clear_confirmation_area()
            self.logger and self.logger.info("❌ Download dibatalkan oleh user")
            if 'error_operation' in self.ui_components:
                self.ui_components['error_operation']("Download dibatalkan")
        
        dialog = create_confirmation_dialog(
            title="📥 Konfirmasi Download Dataset", message=message,
            on_confirm=on_confirm, on_cancel=on_cancel,
            confirm_text="Ya, Download Dataset", cancel_text="Batal"
        )
        
        self._show_confirmation_dialog(dialog)
    
    def _execute_confirmed_download(self, params: Dict[str, Any], completion_callback: Callable = None) -> None:
        """Execute download setelah konfirmasi user."""
        try:
            if self.logger:
                self.logger.info("✅ Parameter valid - memulai download:")
                for key, value in params.items():
                    if key != 'api_key':  # Don't log sensitive data
                        self.logger.info(f"   • {key}: {value}")
            
            # Execute download dengan progress tracking
            result = self.download_service.download_dataset(params)
            
            # Call completion callback
            if completion_callback:
                completion_callback(self.ui_components, result)
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.logger and self.logger.error(f"💥 {error_msg}")
            
            if completion_callback:
                completion_callback(self.ui_components, {'status': 'error', 'message': error_msg})
    
    def _show_confirmation_dialog(self, dialog) -> None:
        """Show confirmation dialog di UI."""
        confirmation_area = self.ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output()
            with confirmation_area:
                display(dialog)
    
    def _clear_confirmation_area(self) -> None:
        """Clear confirmation area."""
        confirmation_area = self.ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output()
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get status execution service untuk debugging."""
        return {
            'download_service_available': self.download_service is not None,
            'confirmation_area_available': 'confirmation_area' in self.ui_components,
            'logger_available': self.logger is not None,
            'ui_components_count': len(self.ui_components),
            'progress_integration': {
                'error_operation': 'error_operation' in self.ui_components,
                'update_progress': 'update_progress' in self.ui_components,
                'complete_operation': 'complete_operation' in self.ui_components
            }
        }