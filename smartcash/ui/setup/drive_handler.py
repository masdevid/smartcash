"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan sinkronisasi dua arah yang ditingkatkan
"""

import os
import shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, List
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

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
    
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            clear_output()
            display(create_info_alert("Menghubungkan ke Google Drive...", "info", ICONS['processing']))
    
    try:
        # Mount drive dan dapatkan path
        drive_path = mount_google_drive(ui_components, silent)
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 1
            ui_components['progress_message'].value = "Status Drive: " + ("terhubung" if drive_path else "tidak terhubung")
        
        if not drive_path:
            # Handle kasus drive tidak terhubung
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = 3
                ui_components['progress_message'].value = "Google Drive gagal terhubung"
            return
        
        # Update status panel dengan create_info_alert
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                clear_output()
                display(create_info_alert("Google Drive berhasil terhubung!", "success", ICONS['success']))
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 2
            ui_components['progress_message'].value = "Membuat struktur direktori di Drive..."
                
        # Pastikan struktur direktori ada di Drive
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                setup_drive_directories(drive_path, ui_components, silent)
        else:
            setup_drive_directories(drive_path, ui_components, silent)
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 3
            ui_components['progress_message'].value = "Membuat symlinks..."
                
        # Buat symlinks
        try:
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    create_symlinks(drive_path, ui_components, silent)
            else:
                create_symlinks(drive_path, ui_components, silent)
        except Exception as e:
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    display(create_info_alert(f"Error saat membuat symlinks: {str(e)}", "warning", ICONS['warning']))
            if logger:
                logger.warning(f"⚠️ Error saat membuat symlinks: {str(e)}")
                
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 4
            ui_components['progress_message'].value = "Sinkronisasi konfigurasi..."
                
        # Sinkronisasi konfigurasi dua arah
        try:
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    sync_configs_bidirectional(drive_path, ui_components, silent)
            else:
                sync_configs_bidirectional(drive_path, ui_components, silent)
        except Exception as e:
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    display(create_info_alert(f"Error saat sinkronisasi konfigurasi: {str(e)}", "warning", ICONS['warning']))
            if logger:
                logger.warning(f"⚠️ Error saat sinkronisasi konfigurasi: {str(e)}")
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = "Proses selesai"
            
        # Jalankan inisialisasi sinkronisasi Drive jika tersedia
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_drive_sync
            initialize_drive_sync(ui_components)
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
        
        # Log success jika logger tersedia
        if logger:
            logger.success(f"✅ Google Drive berhasil terhubung di {drive_path}")
            
    except Exception as e:
        # Update progress pada error
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = f"Error: {str(e)[:30]}..."
            
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                clear_output()
                display(create_info_alert(f"Error saat menghubungkan ke Google Drive: {str(e)}", "error", ICONS['error']))
        
        # Log error jika logger tersedia
        if logger:
            logger.error(f"❌ Error koneksi Google Drive: {str(e)}")

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
            # Update status jika tidak silent
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    display(create_status_indicator("info", f"{ICONS['processing']} Mounting Google Drive..."))
            
            # Mount Drive dengan Google Colab
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Verifikasi mounting berhasil
            time.sleep(1)  # Berikan waktu untuk mounting
            is_mounted, drive_path = detect_drive_mount()
            
            if not is_mounted:
                if not silent and 'status' in ui_components:
                    with ui_components['status']:
                        display(create_info_alert("Gagal mount Google Drive", "error", ICONS['error']))
                return None
        
        # Path dasar Google Drive
        return Path(drive_path)
    except Exception as e:
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_info_alert(f"Error saat mounting Google Drive: {str(e)}", "error", ICONS['error']))
        return None

def setup_drive_directories(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat struktur direktori yang diperlukan di Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Buat direktori SmartCash di Drive jika belum ada
    smartcash_dir = drive_path / 'SmartCash'
    
    # Direktori utama yang diperlukan
    required_dirs = [
        'configs',
        'data', 'data/train', 'data/train/images', 'data/train/labels',
        'data/valid', 'data/valid/images', 'data/valid/labels',
        'data/test', 'data/test/images', 'data/test/labels',
        'runs', 'runs/train', 'runs/train/weights',
        'logs', 'checkpoints'
    ]
    
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator("info", f"{ICONS.get('cleanup', '🧹')} Membuat struktur direktori di Drive..."))
    
    # Buat semua direktori yang diperlukan
    for dir_name in required_dirs:
        dir_path = smartcash_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator("info", f"Direktori dibuat: {dir_path}"))
                
    if logger:
        logger.info(f"✅ Struktur direktori berhasil dibuat di {smartcash_dir}")

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Mapping direktori yang akan dibuat symlink
    smartcash_dir = drive_path / 'SmartCash'
    symlinks = {
        'data': smartcash_dir / 'data',
        'configs': smartcash_dir / 'configs',
        'runs': smartcash_dir / 'runs',
        'logs': smartcash_dir / 'logs',
        'checkpoints': smartcash_dir / 'checkpoints'
    }
    
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator("info", f"🔗 Membuat symlinks..."))
    
    for local_name, target_path in symlinks.items():
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                # Pastikan direktori target ada
                target_path.mkdir(parents=True, exist_ok=True)
                
                local_path = Path(local_name)
                
                # Hapus direktori lokal jika sudah ada
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = local_path.with_name(f"{local_name}_backup")
                    display(create_status_indicator("info", f"Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup"))
                    if backup_path.exists():
                        import shutil
                        shutil.rmtree(backup_path)
                    local_path.rename(backup_path)
                
                # Buat symlink jika belum ada
                if not local_path.exists():
                    local_path.symlink_to(target_path)
                    display(create_status_indicator("success", f"Symlink dibuat: {local_name} → {target_path}"))
                    
    if logger:
        logger.info(f"✅ Symlinks berhasil dibuat ke Google Drive")

def sync_configs_bidirectional(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Sinkronisasi konfigurasi dua arah antara lokal dan Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator("info", f'{ICONS["processing"]} Sinkronisasi Konfigurasi Dua Arah'))
    
    try:
        from smartcash.common.config_sync import sync_all_configs
        
        # Sinkronisasi lanjutan dengan strategi 'merge'
        results = sync_all_configs(
            sync_strategy='merge',
            create_backup=True,
            logger=logger
        )
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                success_count = len(results.get("success", []))
                failure_count = len(results.get("failure", []))
                skipped_count = len(results.get("skipped", []))
                
                summary = []
                if success_count > 0:
                    summary.append(f"{success_count} file disinkronisasi")
                if skipped_count > 0:
                    summary.append(f"{skipped_count} file dilewati")
                
                summary_text = ", ".join(summary)
                
                if failure_count == 0:
                    display(create_status_indicator("success", f"Sinkronisasi selesai: {summary_text}"))
                else:
                    display(create_status_indicator("warning", f"Sinkronisasi selesai dengan peringatan: {summary_text}, {failure_count} gagal"))
        
        if logger:
            logger.info(f"✅ Sinkronisasi konfigurasi selesai: {len(results.get('success', []))} berhasil, {len(results.get('failure', []))} gagal")
    except ImportError:
        # Jika modul config_sync tidak tersedia, gunakan sinkronisasi file biasa
        copy_configs_between_locations(drive_path, ui_components, silent)
        
def copy_configs_between_locations(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Salin file konfigurasi antara lokal dan Google Drive (metode sederhana).
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    local_configs_dir = Path('configs')
    drive_configs_dir = drive_path / 'SmartCash/configs'
    
    # Pastikan direktori ada di kedua lokasi
    local_configs_dir.mkdir(parents=True, exist_ok=True)
    drive_configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Jumlah file yang disalin
    copied_to_drive = 0
    copied_from_drive = 0
    
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator("info", f"🔄 Menyinkronkan file konfigurasi..."))
    
    # Salin file yang ada di lokal tapi tidak ada di drive
    for local_file in local_configs_dir.glob('*.yaml'):
        drive_file = drive_configs_dir / local_file.name
        if not drive_file.exists():
            try:
                shutil.copy2(local_file, drive_file)
                copied_to_drive += 1
                if not silent and 'status' in ui_components:
                    with ui_components['status']:
                        display(create_status_indicator("success", f"File {local_file.name} disalin ke Drive"))
            except Exception as e:
                if logger: logger.warning(f"⚠️ Gagal menyalin {local_file.name} ke Drive: {str(e)}")
    
    # Salin file yang ada di drive tapi tidak ada di lokal
    for drive_file in drive_configs_dir.glob('*.yaml'):
        local_file = local_configs_dir / drive_file.name
        if not local_file.exists():
            try:
                shutil.copy2(drive_file, local_file)
                copied_from_drive += 1
                if not silent and 'status' in ui_components:
                    with ui_components['status']:
                        display(create_status_indicator("success", f"File {drive_file.name} disalin dari Drive"))
            except Exception as e:
                if logger: logger.warning(f"⚠️ Gagal menyalin {drive_file.name} dari Drive: {str(e)}")
    
    # Tampilkan ringkasan
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            summary = []
            if copied_to_drive > 0:
                summary.append(f"{copied_to_drive} file disalin ke Drive")
            if copied_from_drive > 0:
                summary.append(f"{copied_from_drive} file disalin dari Drive")
                
            summary_text = ", ".join(summary) if summary else "Tidak ada file yang perlu disinkronisasi"
            display(create_status_indicator("success" if copied_to_drive + copied_from_drive > 0 else "info", 
                                           f"Sinkronisasi selesai: {summary_text}"))
    
    if logger:
        logger.info(f"✅ Sinkronisasi konfigurasi manual: {copied_to_drive} ke Drive, {copied_from_drive} dari Drive")