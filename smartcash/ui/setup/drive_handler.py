"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan sinkronisasi dua arah yang ditingkatkan
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def setup_drive_handler(ui_components: Dict[str, Any], env=None, config=None, auto_connect: bool = False) -> Dict[str, Any]:
    """
    Setup handler untuk koneksi Google Drive dengan opsi auto-connect.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        auto_connect: Otomatis hubungkan ke Google Drive jika True
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Register handler untuk Drive button
        if 'drive_button' in ui_components and ui_components['drive_button']:
            ui_components['drive_button'].on_click(lambda b: handle_drive_connection(ui_components))
    except Exception as e:
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].warning(f"⚠️ Error setup drive handler: {str(e)}")
    
    # Auto-connect jika diminta
    if auto_connect:
        try:
            handle_drive_connection(ui_components, silent=True)
        except Exception as e:
            if 'logger' in ui_components and ui_components['logger']:
                ui_components['logger'].debug(f"Auto-connect ke Drive gagal: {str(e)}")
    
    return ui_components

def handle_drive_connection(ui_components: Dict[str, Any], silent: bool = False):
    """
    Hubungkan ke Google Drive dan setup struktur proyek dengan integrasi UI utils.
    
    Args:
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            clear_output()
            display(create_info_alert("Menghubungkan ke Google Drive...", "info", ICONS['processing']))
    
    try:
        # Mount drive dan dapatkan path
        drive_path = mount_google_drive(ui_components, silent)
        if not drive_path:
            return
        
        # Update status panel dengan create_info_alert
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                clear_output()
                display(create_info_alert("Google Drive berhasil terhubung!", "success", ICONS['success']))
                
                # Pastikan struktur direktori ada di Drive
                setup_drive_directories(drive_path, ui_components, silent)
                
                # Buat symlinks
                try:
                    create_symlinks(drive_path, ui_components, silent)
                except Exception as e:
                    display(create_info_alert(f"Error saat membuat symlinks: {str(e)}", "warning", ICONS['warning']))
                
                # Sinkronisasi konfigurasi dua arah
                try:
                    sync_configs_bidirectional(drive_path, ui_components, silent)
                except Exception as e:
                    display(create_info_alert(f"Error saat sinkronisasi konfigurasi: {str(e)}", "warning", ICONS['warning']))
        
        # Update panel Colab menggunakan create_info_alert
        if 'colab_panel' in ui_components:
            ui_components['colab_panel'].value = create_info_alert(
                f"""<h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
                <p style="margin:5px 0">{ICONS['success']} Status Google Drive: <strong>terhubung</strong></p>
                <p style="margin:5px 0">Drive terhubung dan struktur direktori telah dibuat.</p>""",
                "success"
            ).value
        
        # Log success jika logger tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].success(f"✅ Google Drive berhasil terhubung di {drive_path}")
            
    except Exception as e:
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                clear_output()
                display(create_info_alert(f"Error saat menghubungkan ke Google Drive: {str(e)}", "error", ICONS['error']))
        
        # Log error jika logger tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].error(f"❌ Error koneksi Google Drive: {str(e)}")

def mount_google_drive(ui_components: Dict[str, Any], silent: bool = False) -> Optional[Path]:
    """
    Mount Google Drive jika belum ter-mount dengan integrasi drive_utils.
    
    Args:
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        Path direktori SmartCash di Google Drive atau None jika gagal
    """
    try:
        # Gunakan utility dari drive_utils
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        
        if not is_mounted:
            from google.colab import drive
            drive.mount('/content/drive')
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
            display(create_status_indicator('info', '📁 Membuat struktur direktori di Drive...'))
    
    # Buat semua direktori yang diperlukan
    for dir_name in required_dirs:
        dir_path = smartcash_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator('info', f"Direktori dibuat: {dir_path}"))

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
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
            display(create_status_indicator('info', '🔗 Membuat symlinks...'))
    
    for local_name, target_path in symlinks.items():
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                # Pastikan direktori target ada
                target_path.mkdir(parents=True, exist_ok=True)
                
                local_path = Path(local_name)
                
                # Hapus direktori lokal jika sudah ada
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = local_path.with_name(f"{local_name}_backup")
                    display(create_status_indicator('info', f"Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup"))
                    if backup_path.exists():
                        import shutil
                        shutil.rmtree(backup_path)
                    local_path.rename(backup_path)
                
                # Buat symlink jika belum ada
                if not local_path.exists():
                    local_path.symlink_to(target_path)
                    display(create_status_indicator('success', f"Symlink dibuat: {local_name} → {target_path}"))

def copy_configs_to_drive(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False) -> List[str]:
    """
    Salin file konfigurasi lokal ke Google Drive jika belum ada di Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        List nama file yang disalin
    """
    local_configs_dir = Path('configs')
    drive_configs_dir = drive_path / 'SmartCash/configs'
    
    # Pastikan direktori ada
    drive_configs_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    if not local_configs_dir.exists():
        # Buat direktori config lokal jika belum ada
        local_configs_dir.mkdir(parents=True, exist_ok=True)
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator('info', f"Direktori configs lokal dibuat"))
                
        return copied_files
    
    # Salin file konfigurasi lokal ke Drive (hanya yang belum ada di Drive)
    for config_file in local_configs_dir.glob('*.yaml'):
        drive_config_file = drive_configs_dir / config_file.name
        
        # Jika file tidak ada di Drive, salin dari lokal ke Drive
        if not drive_config_file.exists():
            shutil.copy2(config_file, drive_config_file)
            copied_files.append(config_file.name)
            
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    display(create_status_indicator('success', f"File konfigurasi {config_file.name} disalin ke Drive"))
    
    return copied_files

def copy_configs_from_drive(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False) -> List[str]:
    """
    Salin file konfigurasi dari Google Drive ke lokal jika belum ada di lokal.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        List nama file yang disalin
    """
    local_configs_dir = Path('configs')
    drive_configs_dir = drive_path / 'SmartCash/configs'
    
    # Pastikan direktori ada
    local_configs_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    if not drive_configs_dir.exists():
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator('info', f"Direktori configs di Drive belum ada"))
                
        return copied_files
    
    # Salin file konfigurasi dari Drive ke lokal (hanya yang belum ada di lokal)
    for config_file in drive_configs_dir.glob('*.yaml'):
        local_config_file = local_configs_dir / config_file.name
        
        # Jika file tidak ada di lokal, salin dari Drive ke lokal
        if not local_config_file.exists():
            shutil.copy2(config_file, local_config_file)
            copied_files.append(config_file.name)
            
            if not silent and 'status' in ui_components:
                with ui_components['status']:
                    display(create_status_indicator('success', f"File konfigurasi {config_file.name} disalin dari Drive"))
    
    return copied_files

def sync_configs_bidirectional(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Sinkronisasi konfigurasi dua arah antara lokal dan Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    if not silent and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator('info', f'{ICONS["processing"]} Sinkronisasi Konfigurasi Dua Arah'))
    
    # Langkah 1: Salin konfigurasi dari lokal ke Drive (jika belum ada di Drive)
    configs_to_drive = copy_configs_to_drive(drive_path, ui_components, silent)
    
    # Langkah 2: Salin konfigurasi dari Drive ke lokal (jika belum ada di lokal)
    configs_from_drive = copy_configs_from_drive(drive_path, ui_components, silent)
    
    # Langkah 3: Sinkronisasi semua config yang sudah ada di kedua tempat
    try:
        from smartcash.common.config_sync import sync_all_configs
        
        # Sinkronisasi lanjutan dengan strategi 'merge' atau 'drive_priority'
        results = sync_all_configs(
            drive_configs_dir=str(drive_path / 'SmartCash/configs'),
            local_configs_dir='configs',
            sync_strategy='merge',  # Gunakan strategi merge untuk konfigurasi yang sudah ada
            create_backup=True
        )
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                success_count = len(results.get("success", []))
                failure_count = len(results.get("failure", []))
                
                summary = []
                if configs_to_drive:
                    summary.append(f"{len(configs_to_drive)} file disalin ke Drive")
                if configs_from_drive:
                    summary.append(f"{len(configs_from_drive)} file disalin dari Drive")
                summary.append(f"{success_count} file disinkronisasi")
                
                summary_text = ", ".join(summary)
                
                if failure_count == 0:
                    display(create_status_indicator('success', f"Sinkronisasi selesai: {summary_text}"))
                else:
                    display(create_status_indicator('warning', f"Sinkronisasi selesai dengan peringatan: {summary_text}, {failure_count} gagal"))
    except ImportError:
        # Jika modul config_sync tidak tersedia, gunakan sinkronisasi file biasa
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                summary = []
                if configs_to_drive:
                    summary.append(f"{len(configs_to_drive)} file disalin ke Drive")
                if configs_from_drive:
                    summary.append(f"{len(configs_from_drive)} file disalin dari Drive")
                
                summary_text = ", ".join(summary) if summary else "Tidak ada file yang perlu disinkronisasi"
                display(create_status_indicator('info', f"Sinkronisasi manual selesai: {summary_text}"))