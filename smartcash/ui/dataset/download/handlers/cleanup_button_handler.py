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
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel'](ui_components, 'info', '🧹 Mempersiapkan pembersihan dataset...')
        
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
    def on_confirm():
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Log konfirmasi
        log_message(ui_components, f"Konfirmasi diterima, menghapus {total_files} file dari {output_dir}", "info", "✅")
        
        # Jalankan proses cleanup
        _execute_cleanup(ui_components)
    
    # Fungsi untuk membatalkan
    def on_cancel():
        # Bersihkan area konfirmasi
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Log pembatalan
        log_message(ui_components, "Pembersihan dibatalkan oleh pengguna", "info", "❌")
        
        # Reset status
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel'](ui_components, 'info', '❌ Pembersihan dibatalkan')
        
        # Aktifkan kembali tombol
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = False
    
    # Pastikan area konfirmasi tersedia
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
        
        # Tambahkan ke UI jika memungkinkan
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                children = list(ui_components['ui'].children)
                children.append(ui_components['confirmation_area'])
                ui_components['ui'].children = tuple(children)
            except Exception as e:
                log_message(ui_components, f"Tidak bisa menambahkan area konfirmasi ke UI: {str(e)}", "warning", "⚠️")
    
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
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](
            ui_components, 
            'warning', 
            '⚠️ Silakan konfirmasi untuk melanjutkan pembersihan'
        )

def _execute_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Eksekusi proses pembersihan hasil download.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan direktori output dari UI
        output_dir = None
        if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
            output_dir = ui_components['output_dir'].value
        elif 'config' in ui_components and isinstance(ui_components['config'], dict):
            output_dir = ui_components['config'].get('data', {}).get('dir', 'data')
        
        if not output_dir:
            output_dir = 'data'  # Default jika tidak ada
        
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
        
        # Update status
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel'](
                ui_components, 
                'info', 
                f'🧹 Membersihkan {total_files} file dari {output_dir}...'
            )
        
        # Tampilkan progress
        if 'log_output' in ui_components:
            with ui_components['log_output']:
                # Gunakan tqdm untuk progress bar
                progress_bar = tqdm(
                    total=total_files,
                    desc=f"🗑️ Menghapus file dari {output_dir}",
                    colour='red',
                    unit='file'
                )
                
                # Hapus file satu per satu dengan progress
                for root, dirs, files in os.walk(output_dir, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            progress_bar.update(1)
                        except Exception as e:
                            log_message(ui_components, f"Gagal menghapus {file_path}: {str(e)}", "error", "❌")
                    
                    # Hapus direktori kosong
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):  # Cek apakah direktori kosong
                                os.rmdir(dir_path)
                        except Exception as e:
                            log_message(ui_components, f"Gagal menghapus direktori {dir_path}: {str(e)}", "error", "❌")
                
                progress_bar.close()
        else:
            # Tanpa progress bar jika tidak ada log_output
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        log_message(ui_components, f"Gagal menghapus {file_path}: {str(e)}", "error", "❌")
                
                # Hapus direktori kosong
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Cek apakah direktori kosong
                            os.rmdir(dir_path)
                    except Exception as e:
                        log_message(ui_components, f"Gagal menghapus direktori {dir_path}: {str(e)}", "error", "❌")
        
        # Proses selesai
        _cleanup_complete(ui_components, success=True, message=f"Berhasil membersihkan {total_files} file dari {output_dir}")
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat membersihkan hasil download: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        _cleanup_complete(ui_components, success=False, message=error_msg)

def _cleanup_complete(ui_components: Dict[str, Any], success: bool, message: str) -> None:
    """
    Selesaikan proses cleanup dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        success: Status keberhasilan cleanup
        message: Pesan hasil cleanup
    """
    # Update status
    status_type = 'success' if success else 'error'
    icon = '✅' if success else '❌'
    
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, status_type, f'{icon} {message}')
    
    # Aktifkan kembali tombol
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].disabled = False
    
    # Log hasil
    log_message(ui_components, message, status_type, icon)
