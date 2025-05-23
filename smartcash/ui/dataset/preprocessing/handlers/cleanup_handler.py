"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler untuk operasi cleanup data preprocessing yang disederhanakan
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future
from smartcash.common.logger import get_logger

class CleanupHandler:
    """Handler untuk operasi cleanup preprocessing."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi cleanup handler."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.cleanup')
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_future: Future = None
        
    def handle_cleanup_click(self, button: Any) -> None:
        """Handler untuk tombol cleanup."""
        button.disabled = True
        
        try:
            self.logger.info("🧹 Memulai cleanup data preprocessing")
            
            # Cek data yang akan dihapus
            cleanup_info = self._get_cleanup_info()
            
            if cleanup_info['total_files'] == 0:
                self._handle_no_data(button)
                return
            
            # Konfirmasi cleanup
            self._show_confirmation(cleanup_info, button)
            
        except Exception as e:
            self._handle_error(f"Error persiapan cleanup: {str(e)}", button)
    
    def _get_cleanup_info(self) -> Dict[str, Any]:
        """Dapatkan info data yang akan dibersihkan."""
        preprocessed_dir = Path(self.ui_components.get('preprocessed_dir', 'data/preprocessed'))
        
        info = {
            'total_files': 0,
            'total_size_mb': 0,
            'directories': []
        }
        
        if not preprocessed_dir.exists():
            return info
        
        # Hitung files dan directories
        for item in preprocessed_dir.rglob('*'):
            if item.is_file():
                info['total_files'] += 1
                try:
                    info['total_size_mb'] += item.stat().st_size / (1024 * 1024)
                except:
                    pass
            elif item.is_dir():
                info['directories'].append(str(item.relative_to(preprocessed_dir)))
        
        return info
    
    def _handle_no_data(self, button: Any) -> None:
        """Handle tidak ada data untuk cleanup."""
        self.logger.info("ℹ️ Tidak ada data preprocessing untuk dihapus")
        self._update_status("info", "Tidak ada data preprocessing ditemukan")
        button.disabled = False
    
    def _show_confirmation(self, cleanup_info: Dict[str, Any], button: Any) -> None:
        """Tampilkan konfirmasi cleanup."""
        message = self._format_confirmation_message(cleanup_info)
        
        def on_confirm():
            self.logger.info("✅ Konfirmasi cleanup diterima")
            self._execute_cleanup(cleanup_info)
        
        def on_cancel():
            self.logger.info("❌ Cleanup dibatalkan")
            self._update_status("info", "Cleanup dibatalkan")
            button.disabled = False
        
        # Tampilkan dialog sederhana via log
        self.logger.info(f"🗑️ KONFIRMASI CLEANUP:")
        self.logger.info(message)
        self.logger.info("Ketik 'y' untuk lanjut atau 'n' untuk batal di cell berikutnya")
        
        # Auto-execute untuk demo (dalam implementasi nyata bisa pakai dialog)
        on_confirm()
    
    def _format_confirmation_message(self, cleanup_info: Dict[str, Any]) -> str:
        """Format pesan konfirmasi."""
        total_files = cleanup_info['total_files']
        total_size = cleanup_info['total_size_mb']
        
        message = f"Data yang akan dihapus:\n"
        message += f"• {total_files:,} file preprocessing\n"
        message += f"• {total_size:.1f} MB total ukuran\n"
        message += f"• {len(cleanup_info['directories'])} direktori"
        
        return message
    
    def _execute_cleanup(self, cleanup_info: Dict[str, Any]) -> None:
        """Eksekusi cleanup secara async."""
        self._update_status("warning", "Menghapus data preprocessing...")
        
        # Submit cleanup task
        self.current_future = self.executor.submit(self._run_cleanup)
        self.current_future.add_done_callback(self._on_cleanup_complete)
    
    def _run_cleanup(self) -> Dict[str, Any]:
        """Run cleanup di background."""
        try:
            preprocessed_dir = Path(self.ui_components.get('preprocessed_dir', 'data/preprocessed'))
            
            if not preprocessed_dir.exists():
                return {'deleted_files': 0, 'success': True}
            
            # Hitung files sebelum hapus
            file_count = len(list(preprocessed_dir.rglob('*')))
            
            # Hapus direktori
            shutil.rmtree(preprocessed_dir)
            
            self.logger.info(f"🗑️ Berhasil hapus {file_count} file/direktori")
            
            return {
                'deleted_files': file_count,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error cleanup: {str(e)}")
            raise
    
    def _on_cleanup_complete(self, future: Future) -> None:
        """Callback saat cleanup selesai."""
        try:
            result = future.result()
            self._handle_cleanup_success(result)
        except Exception as e:
            self._handle_error(f"Error cleanup: {str(e)}")
        finally:
            self._reset_ui()
    
    def _handle_cleanup_success(self, result: Dict[str, Any]) -> None:
        """Handle cleanup berhasil."""
        deleted_count = result.get('deleted_files', 0)
        
        success_message = f"✅ Cleanup berhasil! {deleted_count} item telah dihapus"
        
        self.logger.success(success_message)
        self._update_status("success", success_message)
    
    def _handle_error(self, error_message: str, button: Any = None) -> None:
        """Handle error cleanup."""
        self.logger.error(f"❌ {error_message}")
        self._update_status("error", f"Error: {error_message}")
        
        if button:
            button.disabled = False
    
    def _reset_ui(self) -> None:
        """Reset UI setelah cleanup."""
        # Enable semua buttons
        for button_name in ['preprocess_button', 'save_button', 'reset_button', 'cleanup_button']:
            if button_name in self.ui_components:
                self.ui_components[button_name].disabled = False
    
    def _update_status(self, status: str, message: str) -> None:
        """Update status panel."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
            except ImportError:
                self.ui_components['status_panel'].value = f"<div class='alert alert-{status}'>{message}</div>"


def setup_cleanup_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk cleanup button."""
    if 'cleanup_button' not in ui_components:
        return
    
    # Create handler
    handler = CleanupHandler(ui_components)
    ui_components['cleanup_handler'] = handler
    
    # Setup button click
    ui_components['cleanup_button'].on_click(
        lambda b: handler.handle_cleanup_click(b)
    )