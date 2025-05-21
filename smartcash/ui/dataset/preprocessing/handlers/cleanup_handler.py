"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler untuk operasi pembersihan data preprocessing dataset
"""

from typing import Dict, Any, Optional, List
import os
import shutil
from pathlib import Path
from IPython.display import display

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_ui_state, update_status_panel, reset_after_operation
)
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    start_progress, update_progress, reset_progress_bar, complete_progress
)
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def handle_cleanup_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol pembersihan data preprocessing.
    
    Args:
        button: Tombol yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Get direktori preprocessing
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Cek apakah direktori ada
        if not os.path.exists(preprocessed_dir):
            log_message(ui_components, f"Direktori {preprocessed_dir} tidak ditemukan", "warning", "⚠️")
            update_status_panel(ui_components, "warning", f"Direktori {preprocessed_dir} tidak ditemukan")
            
            # Re-enable tombol
            if button and hasattr(button, 'disabled'):
                button.disabled = False
            return
        
        # Log memulai pembersihan
        log_message(ui_components, f"Memeriksa data preprocessing di {preprocessed_dir}...", "info", "🔄")
        
        # Update UI status
        update_status_panel(ui_components, "info", "Memeriksa data preprocessing...")
        
        # Tampilkan konfirmasi
        confirm_cleanup(ui_components, preprocessed_dir, button)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat persiapan pembersihan: {error_message}")
        log_message(ui_components, f"Error saat persiapan pembersihan: {error_message}", "error", "❌")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def confirm_cleanup(ui_components: Dict[str, Any], dir_path: str, button: Any = None) -> None:
    """
    Tampilkan dialog konfirmasi untuk pembersihan data preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        dir_path: Path direktori yang akan dibersihkan
        button: Tombol yang diklik
    """
    # Pastikan dir_path valid
    path = Path(dir_path)
    
    # Cek apakah direktori ada
    if not path.exists():
        log_message(ui_components, f"Direktori {dir_path} tidak ada", "error", "❌")
        update_status_panel(ui_components, "error", f"Direktori {dir_path} tidak ada")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
            return
    
    # Hitung jumlah file
    file_count = sum(len(files) for _, _, files in os.walk(dir_path))
    
    # Buat pesan konfirmasi
    message = f"Anda akan menghapus semua file preprocessing di direktori {dir_path}. "
    message += f"Terdapat {file_count} file yang akan dihapus. "
    message += "Tindakan ini tidak dapat dibatalkan. "
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Log informasi konfirmasi
    log_message(ui_components, "Menampilkan konfirmasi pembersihan data", "info", "❓")
    
    # Pastikan area konfirmasi visible dan memiliki properti yang tepat
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'layout'):
        ui_components['confirmation_area'].layout.display = 'block'
        ui_components['confirmation_area'].layout.visibility = 'visible'
        ui_components['confirmation_area'].layout.height = 'auto'
        ui_components['confirmation_area'].layout.min_height = '150px'
        ui_components['confirmation_area'].layout.margin = '10px 0'
        ui_components['confirmation_area'].layout.border = '1px solid #ddd'
        ui_components['confirmation_area'].layout.padding = '10px'
    
    # Fungsi untuk menjalankan pembersihan dan membersihkan dialog
    def confirm_and_execute():
        log_message(ui_components, "Konfirmasi pembersihan diterima", "info", "✅")
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Jalankan pembersihan langsung
        execute_cleanup(ui_components, dir_path)
    
    # Fungsi untuk membatalkan pembersihan - Menerima parameter b dari callback
    def cancel_cleanup(b=None):
        log_message(ui_components, "Pembersihan preprocessing dibatalkan", "info", "ℹ️")
        update_status_panel(ui_components, "info", "Pembersihan preprocessing dibatalkan")
        
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    # Bersihkan area konfirmasi terlebih dahulu
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Buat dan tampilkan dialog konfirmasi
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Hapus Data Preprocessing",
            message=message,
            on_confirm=confirm_and_execute,
            on_cancel=cancel_cleanup
        )
        display(dialog)

def execute_cleanup(ui_components: Dict[str, Any], dir_path: str) -> None:
    """
    Eksekusi pembersihan data preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        dir_path: Path direktori yang akan dibersihkan
    """
    # Siapkan UI untuk operasi
    update_status_panel(ui_components, "warning", "Menghapus data preprocessing...")
    
    # Tampilkan progress bar
    start_progress(ui_components, "Membersihkan data preprocessing...")
    
    try:
        # Import notification manager jika diperlukan
        from smartcash.ui.dataset.preprocessing.utils.notification_manager import get_notification_manager
        
        # Get notification manager
        notification_manager = get_notification_manager(ui_components)
        
        # Notify process start
        notification_manager.notify_process_start("cleanup", "Pembersihan dataset preprocessing")
        
        # Update progress
        update_progress(ui_components, 10, 100, "Mengindentifikasi file yang akan dihapus...")
        
        # Jalankan operasi pembersihan
        result = cleanup_preprocessed_files(dir_path, ui_components)
        
        # Log hasil pembersihan
        if result and 'deleted_count' in result:
            deleted_count = result.get('deleted_count', 0)
            log_message(ui_components, f"Berhasil menghapus {deleted_count} file preprocessing", "success", "✅")
            
            # Notify completion
            notification_manager.notify_process_complete({
                'deleted_count': deleted_count,
                'directory': dir_path
            }, "Pembersihan data preprocessing")
            
            # Update UI dengan hasil
            update_status_panel(ui_components, "success", f"Berhasil menghapus {deleted_count} file preprocessing")
        else:
            log_message(ui_components, "Tidak ada file preprocessing yang dihapus", "info", "ℹ️")
            update_status_panel(ui_components, "info", "Tidak ada file preprocessing yang dihapus")
        
        # Selesaikan progress
        complete_progress(ui_components, "Pembersihan selesai")
            
    except Exception as e:
        # Log error
        error_message = f"Error saat membersihkan data preprocessing: {str(e)}"
        log_message(ui_components, error_message, "error", "❌")
        
        # Update UI state
        update_ui_state(ui_components, "error", error_message)
        
        # Reset progress
        reset_progress_bar(ui_components)
        
        # Notify error jika notification manager tersedia
        try:
            notification_manager.notify_process_error(error_message)
        except Exception:
            pass
    
    finally:
        # Reset UI setelah operasi
        reset_ui_after_cleanup(ui_components)

def cleanup_preprocessed_files(dir_path: str, ui_components: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Hapus semua file preprocessing dengan progress reporting.
    
    Args:
        dir_path: Path direktori yang akan dibersihkan
        ui_components: Dictionary komponen UI untuk progress reporting (opsional)
        
    Returns:
        Dictionary statistik hasil pembersihan
    """
    # Inisialisasi statistik
    stats = {
        'deleted_count': 0,
        'failed_count': 0,
        'directories_cleaned': []
    }
    
    # Log function
    def log_func(message, level="info", icon="ℹ️"):
        if ui_components:
            log_message(ui_components, message, level, icon)
        else:
            print(f"{icon} {message}")
    
    try:
        # Cek apakah direktori ada
        path = Path(dir_path)
        if not path.exists():
            log_func(f"Direktori {dir_path} tidak ditemukan", "warning", "⚠️")
            return stats
        
        # Dapatkan daftar semua file yang akan dihapus
        files_to_delete: List[str] = []
        directories_to_clean: List[str] = []
        
        # Scan direktori
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                files_to_delete.append(os.path.join(root, file))
            for dir in dirs:
                directories_to_clean.append(os.path.join(root, dir))
        
        # Urutkan directories dari terdalam ke terluar
        directories_to_clean.sort(key=lambda x: -x.count(os.path.sep))
        
        # Hitung total file
        total_files = len(files_to_delete)
        
        # Jika tidak ada file yang dihapus
        if total_files == 0:
            log_func("Tidak ada file preprocessing yang perlu dihapus", "info", "ℹ️")
            return stats
        
        # Log mulai operasi
        log_func(f"Menghapus {total_files} file preprocessing dari {dir_path}", "info", "🗑️")
        
        # Hapus semua file dengan progress reporting
        for i, file_path in enumerate(files_to_delete):
            try:
                # Update progress jika UI components tersedia
                if ui_components:
                    progress_percent = int((i / total_files) * 80) + 10
                    update_progress(
                        ui_components, 
                        progress_percent, 
                        100, 
                        f"Menghapus file {i+1}/{total_files}",
                        f"File: {os.path.basename(file_path)}"
                    )
                
                # Hapus file
                os.remove(file_path)
                stats['deleted_count'] += 1
                
                # Log setiap 10% dari total file
                if i % max(1, total_files // 10) == 0:
                    log_func(f"Menghapus file {i+1}/{total_files}", "info", "🔄")
                    
            except Exception as e:
                # Log error
                log_func(f"Gagal menghapus {file_path}: {str(e)}", "error", "❌")
                stats['failed_count'] += 1
        
        # Update progress untuk pembersihan direktori
        if ui_components:
            update_progress(ui_components, 90, 100, "Membersihkan direktori kosong...")
            
        # Hapus direktori kosong
        for dir_path in directories_to_clean:
            try:
                # Cek apakah direktori kosong
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    stats['directories_cleaned'].append(dir_path)
            except Exception as e:
                log_func(f"Gagal membersihkan direktori {dir_path}: {str(e)}", "warning", "⚠️")
        
        # Update progress ke 100%
        if ui_components:
            update_progress(ui_components, 100, 100, "Pembersihan selesai", f"Berhasil menghapus {stats['deleted_count']} file")
            
        # Log hasil pembersihan
        log_func(f"Berhasil menghapus {stats['deleted_count']} file dan membersihkan {len(stats['directories_cleaned'])} direktori", "success", "✅")
        
        return stats
        
    except Exception as e:
        # Log error
        error_message = f"Error saat membersihkan data preprocessing: {str(e)}"
        log_func(error_message, "error", "❌")
        
        # Add error to stats
        stats['error'] = error_message
        
        return stats

def reset_ui_after_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI setelah operasi pembersihan selesai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Re-enable cleanup button
    if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'disabled'):
        ui_components['cleanup_button'].disabled = False
    
    # Re-enable tombol lain jika ada
    if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'disabled'):
        ui_components['preprocess_button'].disabled = False
        
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()