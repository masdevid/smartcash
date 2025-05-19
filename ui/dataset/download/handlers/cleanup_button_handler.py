"""
File: smartcash/ui/dataset/download/handlers/cleanup_button_handler.py
Deskripsi: Handler untuk tombol cleanup pada modul download dataset
"""

from typing import Dict, Any, Optional
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm

def handle_cleanup_button_click(b: Any, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol cleanup dataset download.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    
    # Nonaktifkan tombol selama proses
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].disabled = True
    
    # Update status
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, 'info', '🧹 Membersihkan hasil download...')
    
    # Jalankan proses cleanup di thread terpisah
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_execute_cleanup, ui_components)

def _execute_cleanup(ui_components: Dict[str, Any]) -> None:
    """
    Eksekusi proses pembersihan hasil download.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Dapatkan direktori output dari UI atau config
        output_dir = None
        if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
            output_dir = ui_components['output_dir'].value
        elif 'config' in ui_components and isinstance(ui_components['config'], dict):
            output_dir = ui_components['config'].get('data', {}).get('dir', 'data')
        
        if not output_dir:
            output_dir = 'data'  # Default jika tidak ada
        
        # Cek apakah direktori ada
        if not os.path.exists(output_dir):
            if logger:
                logger.warning(f"⚠️ Direktori {output_dir} tidak ditemukan, tidak ada yang perlu dibersihkan")
            _cleanup_complete(ui_components, success=True, message=f"Direktori {output_dir} tidak ditemukan")
            return
        
        # Hitung jumlah file yang akan dihapus
        total_files = sum([len(files) for _, _, files in os.walk(output_dir)])
        
        if total_files == 0:
            if logger:
                logger.info(f"ℹ️ Tidak ada file di {output_dir}, tidak ada yang perlu dibersihkan")
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
                            if logger:
                                logger.error(f"❌ Gagal menghapus {file_path}: {str(e)}")
                    
                    # Hapus direktori kosong
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):  # Cek apakah direktori kosong
                                os.rmdir(dir_path)
                        except Exception as e:
                            if logger:
                                logger.error(f"❌ Gagal menghapus direktori {dir_path}: {str(e)}")
                
                progress_bar.close()
        else:
            # Tanpa progress bar jika tidak ada log_output
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        if logger:
                            logger.error(f"❌ Gagal menghapus {file_path}: {str(e)}")
                
                # Hapus direktori kosong
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Cek apakah direktori kosong
                            os.rmdir(dir_path)
                    except Exception as e:
                        if logger:
                            logger.error(f"❌ Gagal menghapus direktori {dir_path}: {str(e)}")
        
        # Proses selesai
        _cleanup_complete(ui_components, success=True, message=f"Berhasil membersihkan {total_files} file dari {output_dir}")
        
    except Exception as e:
        # Tangani error
        error_msg = f"Error saat membersihkan hasil download: {str(e)}"
        if logger:
            logger.error(f"❌ {error_msg}")
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
    logger = ui_components.get('logger')
    if logger:
        log_func = logger.info if success else logger.error
        log_func(f"{icon} {message}")
