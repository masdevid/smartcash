"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler untuk tombol cleanup pada preprocessing dataset
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_ui_state, update_status_panel, show_confirmation, reset_after_operation
from smartcash.ui.dataset.preprocessing.utils.progress_manager import update_progress, reset_progress_bar
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog

def handle_cleanup_button_click(button: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk klik tombol cleanup preprocessing.
    
    Args:
        button: Widget button yang diklik
        ui_components: Dictionary komponen UI
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Get preprocessed directory
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Log mulai cleanup preprocessing
        log_message(ui_components, "Memulai pembersihan data preprocessing...", "info", "🧹")
        
        # Update UI state
        update_status_panel(ui_components, "warning", "Konfirmasi pembersihan data preprocessing...")
        
        # Tampilkan dialog konfirmasi
        confirm_cleanup(ui_components, preprocessed_dir, button)
        
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat pembersihan: {error_message}")
        log_message(ui_components, f"Error saat persiapan pembersihan preprocessing: {error_message}", "error", "❌")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def confirm_cleanup(ui_components: Dict[str, Any], dir_path: str, button: Any = None) -> None:
    """
    Tampilkan dialog konfirmasi untuk cleanup preprocessing.
    
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
    
    # Buat pesan konfirmasi
    message = f"Anda akan menghapus semua file preprocessing di direktori {dir_path}. "
    message += "Tindakan ini tidak dapat dibatalkan. "
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Fungsi untuk menjalankan cleanup dan membersihkan dialog
    def confirm_and_execute():
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Jalankan cleanup langsung (tanpa threading untuk kompatibilitas dengan Colab)
        execute_cleanup(ui_components, dir_path)
    
    # Fungsi untuk membatalkan cleanup
    def cancel_cleanup(_=None):
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        log_message(ui_components, "Pembersihan preprocessing dibatalkan", "info", "ℹ️")
        update_status_panel(ui_components, "info", "Pembersihan preprocessing dibatalkan")
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    # Gunakan fungsi konfirmasi dari ui_state_manager
    show_confirmation(
        ui_components=ui_components,
        title="Konfirmasi Hapus Data Preprocessing",
        message=message,
        on_confirm=confirm_and_execute,
        on_cancel=cancel_cleanup
    )

def execute_cleanup(ui_components: Dict[str, Any], dir_path: str) -> None:
    """
    Eksekusi cleanup preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        dir_path: Path direktori yang akan dibersihkan
    """
    # Set flag bahwa cleanup sedang berjalan
    ui_components['cleanup_running'] = True
    
    # Update UI state
    update_ui_state(ui_components, "running", "Membersihkan data preprocessing...")
    
    # Disable tombol-tombol
    _disable_buttons(ui_components, True)
    
    # Tampilkan progress
    start_progress(ui_components, "Mempersiapkan pembersihan data preprocessing...")
    
    try:
        # Import cleaner dari modul preprocessing service
        from smartcash.dataset.services.preprocessor.cleaner import PreprocessedCleaner
        
        # Log mulai cleanup
        log_message(ui_components, f"Membersihkan data preprocessing di direktori: {dir_path}", "info", "🧹")
        
        # Update progress
        update_progress(ui_components, 10, 100, "Memulai pembersihan preprocessing...", "Menghapus file...")
        
        # Initialize cleaner
        cleaner = PreprocessedCleaner(
            data_dir=dir_path,
            logger=ui_components.get('logger'),
            observer_manager=ui_components.get('observer_manager')
        )
        
        # Execute cleanup
        result = cleaner.cleanup()
        
        # Check result
        if result and isinstance(result, dict) and result.get('success', False):
            # Get jumlah file yang dihapus
            num_files = result.get('files_removed', 0)
            num_dirs = result.get('dirs_removed', 0)
            space_freed = result.get('space_freed', 0)
            
            # Format space freed
            if space_freed > 1024 * 1024 * 1024:
                space_str = f"{space_freed / (1024 * 1024 * 1024):.2f} GB"
            elif space_freed > 1024 * 1024:
                space_str = f"{space_freed / (1024 * 1024):.2f} MB"
            elif space_freed > 1024:
                space_str = f"{space_freed / 1024:.2f} KB"
            else:
                space_str = f"{space_freed} bytes"
            
            # Update progress
            update_progress(ui_components, 100, 100, "Pembersihan selesai", f"Dihapus {num_files} files dan {num_dirs} direktori")
            
            # Update UI state
            update_ui_state(ui_components, "success", "Pembersihan selesai")
            
            # Log hasil
            log_message(ui_components, f"Pembersihan preprocessing berhasil: {num_files} file dan {num_dirs} direktori dihapus ({space_str})", "success", "✅")
        else:
            # Cleanup failed
            update_progress(ui_components, 100, 100, "Pembersihan gagal", "Terjadi kesalahan")
            update_ui_state(ui_components, "error", "Pembersihan gagal")
            
            error_message = result.get('error', 'Tidak ada detail error') if isinstance(result, dict) else 'Unknown error'
            log_message(ui_components, f"Pembersihan preprocessing gagal: {error_message}", "error", "❌")
    
    except Exception as e:
        # Log error
        error_message = str(e)
        update_ui_state(ui_components, "error", f"Error saat pembersihan: {error_message}")
        log_message(ui_components, f"Error saat pembersihan preprocessing: {error_message}", "error", "❌")
    
    finally:
        # Reset UI
        reset_ui_after_cleanup(ui_components)

def start_progress(ui_components: Dict[str, Any], message: str = "Memulai pembersihan...") -> None:
    """
    Memulai progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan awal progress
    """
    # Reset progress dulu
    reset_progress_bar(ui_components)
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'visible'
        ui_components['progress_container'].layout.display = 'block'
    
    # Update progress awal
    update_progress(ui_components, 0, 100, message)
    
    # Pastikan log accordion terbuka
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0

def _disable_buttons(ui_components: Dict[str, Any], disable: bool) -> None:
    """
    Disable atau enable tombol-tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
        disable: True untuk disable, False untuk enable
    """
    # Disable atau enable tombol
    for key in ['preprocess_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button']:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disable

def reset_ui_after_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI setelah cleanup selesai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Enable tombol
    _disable_buttons(ui_components, False)
    
    # Disable stop button
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'disabled'):
        ui_components['stop_button'].disabled = True
    
    # Reset flag
    ui_components['cleanup_running'] = False
    
    # Update status
    update_status_panel(ui_components, "success", "Pembersihan selesai")

def cleanup_preprocessed_files(dir_path: str) -> Dict[str, Any]:
    """
    Bersihkan file preprocessing di direktori.
    
    Args:
        dir_path: Path direktori yang akan dibersihkan
        
    Returns:
        Dictionary hasil cleanup
    """
    try:
        # Pastikan path valid
        path = Path(dir_path)
        if not path.exists():
            return {"success": False, "error": f"Direktori {dir_path} tidak ada"}
        
        # Cari semua file dan direktori
        files_removed = 0
        dirs_removed = 0
        
        # Hitung ukuran direktori sebelum dihapus
        total_size = get_dir_size(path)
        
        # Hapus semua subdirektori dan file
        for item in path.glob("**/*"):
            if item.is_file():
                item.unlink()
                files_removed += 1
            elif item.is_dir() and item != path:
                if not any(item.iterdir()):  # Hapus hanya direktori kosong
                    item.rmdir()
                    dirs_removed += 1
        
        # Hapus direktori utama jika kosong
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
            dirs_removed += 1
        
        return {
            "success": True,
            "files_removed": files_removed,
            "dirs_removed": dirs_removed,
            "space_freed": total_size
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_dir_size(path: Path) -> int:
    """
    Dapatkan ukuran direktori dalam bytes.
    
    Args:
        path: Path direktori
        
    Returns:
        Ukuran direktori dalam bytes
    """
    total_size = 0
    
    # Iterasi semua file dalam direktori
    for item in path.glob("**/*"):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size