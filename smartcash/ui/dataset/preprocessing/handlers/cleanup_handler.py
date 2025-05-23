"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler untuk operasi cleanup data preprocessing dengan integrasi service dan dialog confirmation
"""

from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.dialog_utils import show_cleanup_confirmation
from smartcash.ui.dataset.preprocessing.services.cleanup_service import CleanupService
from smartcash.ui.dataset.preprocessing.utils.progress_tracker import ProgressTracker

class CleanupHandler:
    """Handler untuk operasi cleanup dengan confirmation dan progress tracking."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi cleanup handler dengan komponen UI."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.cleanup')
        self.cleanup_service = CleanupService(ui_components)
        self.progress_tracker = ProgressTracker(ui_components, "cleanup")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.cleanup_future: Optional[Future] = None
    
    def handle_cleanup_click(self, button: Any) -> None:
        """Handler utama untuk tombol cleanup."""
        if button and hasattr(button, 'disabled'):
            button.disabled = True
        
        try:
            self.ui_components['cleanup_running'] = True
            self.logger.info("🧹 Memulai proses cleanup data preprocessing")
            
            cleanup_info = self.cleanup_service.get_cleanup_info()
            
            if cleanup_info['total_files'] == 0:
                self._handle_no_data_to_cleanup(button)
                return
            
            self._show_confirmation_dialog(cleanup_info, button)
            
        except Exception as e:
            self._handle_cleanup_error(f"Error saat persiapan cleanup: {str(e)}", button)
    
    def _handle_no_data_to_cleanup(self, button: Any) -> None:
        """Handle kasus tidak ada data untuk dihapus."""
        self.logger.info("ℹ️ Tidak ada data preprocessing yang perlu dihapus")
        self._update_status_panel("info", "Tidak ada data preprocessing yang ditemukan")
        self._reset_cleanup_state(button)
    
    def _show_confirmation_dialog(self, cleanup_info: Dict[str, Any], button: Any) -> None:
        """Tampilkan dialog konfirmasi cleanup."""
        message = self._format_confirmation_message(cleanup_info)
        
        def on_confirm():
            self.logger.info("✅ Konfirmasi cleanup diterima, memulai penghapusan")
            self._execute_cleanup_async()
        
        def on_cancel():
            self.logger.info("❌ Cleanup dibatalkan oleh pengguna")
            self._update_status_panel("info", "Cleanup dibatalkan")
            self._reset_cleanup_state(button)
        
        # FIXED: Use correct function name
        show_cleanup_confirmation(
            self.ui_components,
            message,
            cleanup_info,
            on_confirm,
            on_cancel
        )
    
    def _format_confirmation_message(self, cleanup_info: Dict[str, Any]) -> str:
        """Format pesan konfirmasi berdasarkan info cleanup."""
        total_files = cleanup_info['total_files']
        total_size = cleanup_info.get('total_size_mb', 0)
        directories = cleanup_info.get('directories', [])
        
        message = f"🗑️ **Konfirmasi Penghapusan Data Preprocessing**\n\n"
        message += f"Data yang akan dihapus:\n"
        message += f"• **{total_files}** file preprocessing\n"
        message += f"• **{total_size:.1f} MB** total ukuran\n"
        message += f"• **{len(directories)}** direktori\n\n"
        
        message += "📂 **Direktori yang akan dibersihkan:**\n"
        for directory in directories[:5]:
            message += f"  - {directory}\n"
        
        if len(directories) > 5:
            message += f"  - ... dan {len(directories) - 5} direktori lainnya\n"
        
        message += "\n⚠️ **Peringatan:**\n"
        message += "• Tindakan ini **tidak dapat dibatalkan**\n"
        message += "• Data yang dihapus **tidak dapat dipulihkan**\n"
        message += "• Proses mungkin membutuhkan waktu beberapa menit\n\n"
        message += "Apakah Anda yakin ingin melanjutkan?"
        
        return message
    
    def _execute_cleanup_async(self) -> None:
        """Eksekusi cleanup secara asinkron dengan progress tracking."""
        self.progress_tracker.start("Memulai pembersihan data preprocessing...")
        self._update_status_panel("warning", "Menghapus data preprocessing...")
        
        self.cleanup_future = self.executor.submit(self._run_cleanup_with_progress)
        self.cleanup_future.add_done_callback(self._on_cleanup_complete)
    
    def _run_cleanup_with_progress(self) -> Dict[str, Any]:
        """Jalankan cleanup dengan progress callback."""
        try:
            def progress_callback(progress: int, total: int, message: str, step_message: str = ""):
                self.progress_tracker.update(progress, total, message, step_message)
                
                if self.ui_components.get('cleanup_stopped', False):
                    return False
                
                return True
            
            result = self.cleanup_service.execute_cleanup(progress_callback)
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error saat cleanup: {str(e)}")
            raise
    
    def _on_cleanup_complete(self, future: Future) -> None:
        """Callback yang dipanggil saat cleanup selesai."""
        try:
            result = future.result()
            
            if result.get('success', False):
                self._handle_cleanup_success(result)
            else:
                self._handle_cleanup_error(result.get('error', 'Unknown error'))
                
        except Exception as e:
            self._handle_cleanup_error(f"Error saat mengambil hasil cleanup: {str(e)}")
        
        finally:
            self._reset_cleanup_state()
    
    def _handle_cleanup_success(self, result: Dict[str, Any]) -> None:
        """Handle cleanup yang berhasil."""
        deleted_count = result.get('deleted_files', 0)
        deleted_size = result.get('deleted_size_mb', 0)
        
        success_message = f"✅ Cleanup berhasil! {deleted_count} file ({deleted_size:.1f} MB) telah dihapus"
        
        self.logger.success(success_message)
        self._update_status_panel("success", success_message)
        self.progress_tracker.complete("Pembersihan data selesai")
        self._clear_confirmation_area()
    
    def _handle_cleanup_error(self, error_message: str, button: Any = None) -> None:
        """Handle error saat cleanup."""
        self.logger.error(f"❌ {error_message}")
        self._update_status_panel("error", f"Error cleanup: {error_message}")
        self.progress_tracker.reset()
        self._reset_cleanup_state(button)
    
    def _update_status_panel(self, status: str, message: str) -> None:
        """Update status panel UI."""
        if 'status_panel' in self.ui_components:
            try:
                # Try to use existing status panel update mechanism
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
            except ImportError:
                # Fallback manual update
                status_classes = {
                    'success': 'success',
                    'error': 'danger',
                    'warning': 'warning',
                    'info': 'info'
                }
                css_class = status_classes.get(status, 'info')
                self.ui_components['status_panel'].value = f"<div class='alert alert-{css_class}'>{message}</div>"
    
    def _clear_confirmation_area(self) -> None:
        """Bersihkan area konfirmasi."""
        if 'confirmation_area' in self.ui_components:
            confirmation_area = self.ui_components['confirmation_area']
            if hasattr(confirmation_area, 'clear_output'):
                confirmation_area.clear_output()
            if hasattr(confirmation_area, 'layout'):
                confirmation_area.layout.display = 'none'
    
    def _reset_cleanup_state(self, button: Any = None) -> None:
        """Reset state cleanup ke kondisi awal."""
        self.ui_components['cleanup_running'] = False
        self.ui_components['cleanup_stopped'] = False
        
        if button and hasattr(button, 'disabled'):
            button.disabled = False
        
        button_keys = ['cleanup_button', 'preprocess_button', 'save_button', 'reset_button']
        for button_key in button_keys:
            if button_key in self.ui_components and hasattr(self.ui_components[button_key], 'disabled'):
                self.ui_components[button_key].disabled = False
    
    def stop_cleanup(self) -> None:
        """Hentikan proses cleanup yang sedang berjalan."""
        if self.cleanup_future and not self.cleanup_future.done():
            self.logger.warning("⏹️ Menghentikan proses cleanup...")
            
            self.ui_components['cleanup_stopped'] = True
            
            try:
                self.cleanup_future.cancel()
            except Exception as e:
                self.logger.warning(f"⚠️ Tidak dapat membatalkan cleanup future: {str(e)}")
            
            self._update_status_panel("warning", "Cleanup dihentikan oleh pengguna")
            self.progress_tracker.reset()
            self._reset_cleanup_state()
    
    def cleanup_resources(self) -> None:
        """Cleanup resources saat handler tidak digunakan lagi."""
        try:
            if self.cleanup_future and not self.cleanup_future.done():
                self.stop_cleanup()
            
            self.executor.shutdown(wait=False)
            self.logger.debug("🧹 Cleanup handler resources berhasil dibersihkan")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat cleanup resources: {str(e)}")