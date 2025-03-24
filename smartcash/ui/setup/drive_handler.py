"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan UI logger terintegrasi
"""

import os, shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, List
from IPython.display import display, HTML, clear_output
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.utils.ui_logger import log_to_ui

def handle_drive_connection(ui_components: Dict[str, Any], silent: bool = False):
    """
    Hubungkan ke Google Drive dan setup struktur proyek dengan integrasi UI utils.
    
    Args:
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Update progress tracking
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Memeriksa Google Drive..."
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
    
    # Log ke UI langsung 
    log_to_ui(ui_components, "Menghubungkan ke Google Drive...", "info", ICONS['processing'])
    
    try:
        # Mount drive dan dapatkan path - jalankan dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            drive_result = executor.submit(mount_google_drive, ui_components, silent)
            drive_path = drive_result.result()
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 1
            ui_components['progress_message'].value = "Status Drive: " + ("terhubung" if drive_path else "tidak terhubung")
        
        if not drive_path:
            # Handle kasus drive tidak terhubung
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = 3
                ui_components['progress_message'].value = "Google Drive gagal terhubung"
            
            log_to_ui(ui_components, "Google Drive gagal terhubung", "error", "❌")
                
            # Reset progress
            if hasattr(ui_components, 'reset_progress') and callable(ui_components['reset_progress']):
                ui_components['reset_progress']()
            return
        
        # Update ke UI dengan log_to_ui
        log_to_ui(ui_components, "Google Drive berhasil terhubung!", "success", ICONS['success'])
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 2
            ui_components['progress_message'].value = "Membuat struktur direktori di Drive..."
                
        # Pastikan struktur direktori ada di Drive dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            setup_future = executor.submit(setup_drive_directories, drive_path, ui_components, silent)
            setup_future.result()
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 3
            ui_components['progress_message'].value = "Membuat symlinks..."
                
        # Buat symlinks dengan ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                symlink_future = executor.submit(create_symlinks, drive_path, ui_components, silent)
                symlink_future.result()
        except Exception as e:
            log_to_ui(ui_components, f"Error saat membuat symlinks: {str(e)}", "warning", ICONS['warning'])
            if logger:
                logger.warning(f"⚠️ Error saat membuat symlinks: {str(e)}")
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 4
            ui_components['progress_message'].value = "Sinkronisasi konfigurasi..."
                
        # Sinkronisasi konfigurasi dua arah dengan ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                sync_future = executor.submit(sync_configs_bidirectional, drive_path, ui_components, silent)
                sync_future.result()
        except Exception as e:
            log_to_ui(ui_components, f"Error saat sinkronisasi konfigurasi: {str(e)}", "warning", ICONS['warning'])
            if logger:
                logger.warning(f"⚠️ Error saat sinkronisasi konfigurasi: {str(e)}")
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = "Proses selesai"
            
        # Jalankan inisialisasi sinkronisasi Drive jika tersedia
        try:
            from smartcash.common.drive_sync_initializer import initialize_configs
            success, message = initialize_configs(logger)
            log_to_ui(ui_components, f"🔄 Sinkronisasi konfigurasi: {message}", "info")
        except ImportError:
            pass
        
        # Update panel Colab menggunakan create_info_alert
        if 'colab_panel' in ui_components:
            ui_components['colab_panel'].value = create_info_alert(
                f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">{ICONS['success']} Status Google Drive: <strong>terhubung</strong></p>
                <p style="margin:5px 0">Drive terhubung dan struktur direktori telah dibuat.</p>""",
                "success"
            ).value
            
            # Sembunyikan tombol drive karena sudah terhubung
            ui_components['drive_button'].layout.display = 'none'
        
        # Log success dengan UI logger
        log_to_ui(ui_components, f"Google Drive berhasil terhubung di {drive_path}", "success", "✅")
            
        # Reset progress setelah selesai
        if hasattr(ui_components, 'reset_progress') and callable(ui_components['reset_progress']):
            ui_components['reset_progress']()
            
    except Exception as e:
        # Update progress pada error
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = f"Error: {str(e)[:30]}..."
        
        # Log error ke UI logger
        log_to_ui(ui_components, f"Error saat menghubungkan ke Google Drive: {str(e)}", "error", ICONS['error'])
        
        # Reset progress setelah error
        if hasattr(ui_components, 'reset_progress') and callable(ui_components['reset_progress']):
            ui_components['reset_progress']()

def mount_google_drive(ui_components: Dict[str, Any], silent: bool = False) -> Optional[Path]:
    """
    Mount Google Drive jika belum ter-mount dengan integrasi drive_utils.
    
    Args:
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        Path direktori SmartCash di Google Drive atau None jika gagal
    """
    logger = ui_components.get('logger')
    
    try:
        # Gunakan utility dari drive_utils
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        
        if not is_mounted:
            # Update status dengan log_to_ui
            log_to_ui(ui_components, f"Mounting Google Drive...", "info", ICONS['processing'])
            
            # Mount Drive dengan Google Colab
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Verifikasi mounting berhasil
            time.sleep(1)  # Berikan waktu untuk mounting
            is_mounted, drive_path = detect_drive_mount()
            
            if not is_mounted:
                log_to_ui(ui_components, "Gagal mount Google Drive", "error", ICONS['error'])
                return None
        
        # Path dasar Google Drive
        base_path = Path(drive_path)
        
        # Pastikan direktori SmartCash ada
        smartcash_dir = base_path / 'SmartCash'
        smartcash_dir.mkdir(parents=True, exist_ok=True)
        
        return smartcash_dir
    except Exception as e:
        log_to_ui(ui_components, f"Error saat mounting Google Drive: {str(e)}", "error", ICONS['error'])
        return None

def setup_drive_directories(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat struktur direktori yang diperlukan di Google Drive dengan one-liner.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Direktori utama yang diperlukan - konsolisasi one-liner dengan list comprehension
    [(drive_path / dir_name).mkdir(parents=True, exist_ok=True) for dir_name in [
        'configs',
        'data', 'data/train', 'data/train/images', 'data/train/labels',
        'data/valid', 'data/valid/images', 'data/valid/labels',
        'data/test', 'data/test/images', 'data/test/labels',
        'runs', 'runs/train', 'runs/train/weights',
        'logs', 'checkpoints'
    ]]
    
    # Log ke UI
    log_to_ui(ui_components, f"Struktur direktori berhasil dibuat di {drive_path}", "success", ICONS.get('success', '✅'))
                
    if logger:
        logger.info(f"✅ Struktur direktori berhasil dibuat di {drive_path}")

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive dengan one-liner.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Mapping direktori yang akan dibuat symlink
    symlinks = {
        'data': drive_path / 'data',
        'configs': drive_path / 'configs',
        'runs': drive_path / 'runs',
        'logs': drive_path / 'logs',
        'checkpoints': drive_path / 'checkpoints'
    }
    
    log_to_ui(ui_components, "🔗 Membuat symlinks...", "info")
    
    # Buat symlinks dengan one-liner
    for local_name, target_path in symlinks.items():
        # Pastikan direktori target ada
        target_path.mkdir(parents=True, exist_ok=True)
        local_path = Path(local_name)
        
        # Handle existing directory
        if local_path.exists() and not local_path.is_symlink():
            backup_path = local_path.with_name(f"{local_name}_backup")
            log_to_ui(ui_components, f"Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup", "info")
            
            # Hapus backup lama jika ada dan pindahkan folder saat ini ke backup
            if backup_path.exists(): shutil.rmtree(backup_path)
            local_path.rename(backup_path)
        
        # Buat symlink jika belum ada
        if not local_path.exists():
            local_path.symlink_to(target_path)
            log_to_ui(ui_components, f"Symlink dibuat: {local_name} → {target_path}", "success")
                    
    if logger:
        logger.info(f"✅ Symlinks berhasil dibuat ke Google Drive")

def sync_configs_bidirectional(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Sinkronisasi konfigurasi dua arah antara lokal dan Google Drive menggunakan config_sync.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    log_to_ui(ui_components, f'Sinkronisasi Konfigurasi Dua Arah', "info", ICONS["processing"])
    
    try:
        # Coba gunakan modul terpadu config_sync dengan error handling
        try:
            from smartcash.common.config_sync import sync_all_configs
            
            # Sinkronisasi lanjutan dengan strategi 'merge'
            results = sync_all_configs(
                sync_strategy='merge',
                create_backup=True,
                logger=logger
            )
            
            success_count = len(results.get("success", []))
            failure_count = len(results.get("failure", []))
            skipped_count = len(results.get("skipped", []))
            
            log_to_ui(ui_components, 
                f"Sinkronisasi selesai: {success_count} disinkronisasi, {skipped_count} dilewati, {failure_count} gagal",
                "success" if failure_count == 0 else "warning"
            )
            
            if logger:
                logger.info(f"✅ Sinkronisasi konfigurasi selesai: {success_count} berhasil, {failure_count} gagal")
                
        except ImportError:
            # Jika modul tidak tersedia, gunakan implementasi minimal
            log_to_ui(ui_components, "Menggunakan metode sinkronisasi file sederhana", "info")
            copy_configs_between_locations(drive_path, ui_components, silent)
    except Exception as e:
        log_to_ui(ui_components, f"Error saat sinkronisasi: {str(e)}", "error")
        
def copy_configs_between_locations(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Salin file konfigurasi antara lokal dan Google Drive dengan ThreadPoolExecutor.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    local_configs_dir = Path('configs')
    drive_configs_dir = drive_path / 'configs'
    
    # Pastikan direktori ada di kedua lokasi
    local_configs_dir.mkdir(parents=True, exist_ok=True)
    drive_configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Gunakan ThreadPoolExecutor untuk operasi file
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Salin file yang ada di lokal tapi tidak ada di drive
        local_to_drive_tasks = [(local_file, drive_configs_dir / local_file.name) 
                              for local_file in local_configs_dir.glob('*.yaml') 
                              if not (drive_configs_dir / local_file.name).exists()]
        
        # Salin file yang ada di drive tapi tidak ada di lokal
        drive_to_local_tasks = [(drive_file, local_configs_dir / drive_file.name)
                               for drive_file in drive_configs_dir.glob('*.yaml')
                               if not (local_configs_dir / drive_file.name).exists()]
        
        # Helper function untuk menyalin file
        def copy_file_with_logging(src, dst):
            try:
                shutil.copy2(src, dst)
                return {'success': True, 'src': src, 'dst': dst}
            except Exception as e:
                return {'success': False, 'src': src, 'dst': dst, 'error': str(e)}
        
        # Jalankan tugas secara paralel
        local_to_drive_results = list(executor.map(lambda t: copy_file_with_logging(t[0], t[1]), local_to_drive_tasks))
        drive_to_local_results = list(executor.map(lambda t: copy_file_with_logging(t[0], t[1]), drive_to_local_tasks))
    
    # Hitung statistik
    copied_to_drive = sum(1 for r in local_to_drive_results if r['success'])
    copied_from_drive = sum(1 for r in drive_to_local_results if r['success'])
    
    # Tampilkan hasil
    if copied_to_drive > 0:
        log_to_ui(ui_components, f"{copied_to_drive} file disalin ke Drive", "success")
    if copied_from_drive > 0:
        log_to_ui(ui_components, f"{copied_from_drive} file disalin dari Drive", "success")
    if copied_to_drive == 0 and copied_from_drive == 0:
        log_to_ui(ui_components, "Tidak ada file yang perlu disinkronisasi", "info")
    
    if logger:
        logger.info(f"✅ Sinkronisasi konfigurasi manual: {copied_to_drive} ke Drive, {copied_from_drive} dari Drive")