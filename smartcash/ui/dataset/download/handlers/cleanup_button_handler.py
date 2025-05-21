"""
File: smartcash/ui/dataset/download/handlers/cleanup_button_handler.py
Deskripsi: Handler untuk tombol cleanup pada modul download dataset
"""

from typing import Dict, Any, Optional
import os
import shutil
from tqdm.notebook import tqdm
from IPython.display import display
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import update_status_panel, enable_download_button, ensure_confirmation_area
from smartcash.ui.dataset.download.utils.progress_manager import show_progress, update_progress

def handle_cleanup_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol cleanup dataset download.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Nonaktifkan tombol selama proses
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].disabled = True
    
    try:
        # Update status
        update_status_panel(ui_components, '🧹 Mempersiapkan pembersihan dataset...', 'info')
        
        # Dapatkan direktori output dari UI
        output_dir = None
        if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
            output_dir = ui_components['output_dir'].value
        elif 'config' in ui_components and isinstance(ui_components['config'], dict):
            output_dir = ui_components['config'].get('data', {}).get('dir', 'data')
        
        if not output_dir:
            output_dir = 'data'  # Default jika tidak ada
            
        # Log message
        log_message(ui_components, f"Mempersiapkan pembersihan dataset di {output_dir}", "info", "🧹")
        
        # Cek apakah direktori ada
        if not os.path.exists(output_dir):
            log_message(ui_components, f"Direktori {output_dir} tidak ditemukan, tidak ada yang perlu dibersihkan", "warning", "⚠️")
            _cleanup_complete(ui_components, success=True, message=f"Direktori {output_dir} tidak ditemukan")
            return
        
        # Hitung jumlah file yang akan dihapus
        total_files = sum([len(files) for _, _, files in os.walk(output_dir)])
        
        if total_files == 0:
            log_message(ui_components, f"Tidak ada file di {output_dir}, tidak ada yang perlu dibersihkan", "info", "ℹ️")
            _cleanup_complete(ui_components, success=True, message=f"Tidak ada file di {output_dir}")
            return
        
        # Tampilkan konfirmasi untuk cleanup
        _show_confirmation_dialog(ui_components, output_dir, total_files)
            
    except Exception as e:
        # Tangani error
        log_message(ui_components, f"Error saat persiapan cleanup: {str(e)}", "error", "❌")
        _cleanup_complete(ui_components, success=False, message=f"Error: {str(e)}")

def _show_confirmation_dialog(ui_components: Dict[str, Any], output_dir: str, total_files: int) -> None:
    """
    Tampilkan dialog konfirmasi untuk cleanup.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori yang akan dibersihkan
        total_files: Jumlah file yang akan dihapus
    """
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    # Buat pesan konfirmasi
    message = f"Anda akan menghapus {total_files} file dari direktori {output_dir}.\n"
    message += "Tindakan ini tidak dapat dibatalkan.\n"
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Fungsi untuk menjalankan cleanup setelah konfirmasi
    def on_confirm(b):
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Log konfirmasi
        log_message(ui_components, f"Konfirmasi diterima, menghapus {total_files} file dari {output_dir}", "info", "✅")
        
        # Jalankan proses cleanup
        _execute_cleanup(ui_components, output_dir, total_files)
    
    # Fungsi untuk membatalkan
    def on_cancel(b):
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Log pembatalan
        log_message(ui_components, "Pembersihan dibatalkan oleh pengguna", "info", "❌")
        
        # Reset status
        update_status_panel(ui_components, '❌ Pembersihan dibatalkan', 'info')
        
        # Aktifkan kembali tombol
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = False
    
    # Pastikan area konfirmasi tersedia
    ui_components = ensure_confirmation_area(ui_components)
    
    # Tampilkan dialog konfirmasi
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Pembersihan Dataset", 
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel
        )
        display(dialog)
    
    # Update status panel
    update_status_panel(ui_components, '⚠️ Silakan konfirmasi untuk melanjutkan pembersihan', 'warning')

def _execute_cleanup(ui_components: Dict[str, Any], output_dir: str, total_files: int) -> None:
    """
    Eksekusi proses pembersihan hasil download.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori yang akan dibersihkan
        total_files: Jumlah file yang akan dihapus
    """
    try:
        # Update status
        update_status_panel(ui_components, f'🧹 Membersihkan {total_files} file dari {output_dir}...', 'info')
        
        # Tampilkan progress
        show_progress(ui_components, f"Membersihkan {total_files} file dari {output_dir}...")
        
        # Hapus file dengan progress
        files_cleaned = 0
        
        # Gunakan tqdm untuk progress bar
        with tqdm(total=total_files, desc=f"🗑️ Menghapus file", colour='red', unit='file') as progress_bar:
            # Hapus file satu per satu dengan progress
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Hapus file
                        os.remove(file_path)
                        files_cleaned += 1
                        
                        # Update progress
                        progress_bar.update(1)
                        progress_bar.set_description(f"🗑️ Menghapus {file}")
                        
                        # Update progress UI jika tersedia
                        progress_percent = min(int((files_cleaned / total_files) * 100), 100)
                        update_progress(ui_components, progress_percent, f"Membersihkan file {files_cleaned}/{total_files}...")
                        
                    except Exception as e:
                        log_message(ui_components, f"Gagal menghapus {file_path}: {str(e)}", "warning", "⚠️")
            
            # Hapus direktori kosong
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        # Cek apakah direktori kosong
                        if not os.listdir(dir_path):
                            # Hapus direktori kosong
                            os.rmdir(dir_path)
                            log_message(ui_components, f"Menghapus direktori kosong: {dir_path}", "debug", "🗑️")
                    except Exception as e:
                        log_message(ui_components, f"Gagal menghapus direktori {dir_path}: {str(e)}", "warning", "⚠️")
        
        # Update progress ke 100%
        update_progress(ui_components, 100, "Pembersihan selesai")
        
        # Log selesai
        log_message(ui_components, f"Berhasil membersihkan {files_cleaned} file dari {output_dir}", "success", "✅")
        
        # Update status complete
        _cleanup_complete(ui_components, success=True, message=f"Berhasil membersihkan {files_cleaned} file")
        
    except Exception as e:
        # Tangani error
        log_message(ui_components, f"Error saat membersihkan direktori: {str(e)}", "error", "❌")
        _cleanup_complete(ui_components, success=False, message=f"Error: {str(e)}")

def _cleanup_complete(ui_components: Dict[str, Any], success: bool, message: str) -> None:
    """
    Selesaikan proses cleanup dan reset UI.
    
    Args:
        ui_components: Dictionary komponen UI
        success: Status keberhasilan
        message: Pesan status
    """
    # Update status berdasarkan hasil
    if success:
        update_status_panel(ui_components, f"✅ {message}", "success")
    else:
        update_status_panel(ui_components, f"❌ {message}", "error")
    
    # Aktifkan kembali tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].disabled = False
    
    # Reset flag cleanup_running jika ada
    if 'cleanup_running' in ui_components:
        ui_components['cleanup_running'] = False
