"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan integrasi UI utils dan auto-connect
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
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
                
                # Buat symlinks
                try:
                    create_symlinks(drive_path, ui_components, silent)
                except Exception as e:
                    display(create_info_alert(f"Error saat membuat symlinks: {str(e)}", "warning", ICONS['warning']))
                
                # Sinkronisasi konfigurasi
                try:
                    sync_configs(drive_path, ui_components, silent)
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
        
        # Buat direktori SmartCash di Drive jika belum ada
        smartcash_dir = Path(drive_path) / 'SmartCash'
        os.makedirs(smartcash_dir, exist_ok=True)
        os.makedirs(smartcash_dir / 'configs', exist_ok=True)
        os.makedirs(smartcash_dir / 'data', exist_ok=True)
        os.makedirs(smartcash_dir / 'runs', exist_ok=True)
        os.makedirs(smartcash_dir / 'logs', exist_ok=True)
        
        return smartcash_dir
    except Exception as e:
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_info_alert(f"Error saat mounting Google Drive: {str(e)}", "error", ICONS['error']))
        return None

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    # Mapping direktori yang akan dibuat symlink
    symlinks = {
        'data': drive_path / 'data',
        'configs': drive_path / 'configs',
        'runs': drive_path / 'runs',
        'logs': drive_path / 'logs',
        'checkpoints': drive_path / 'checkpoints'
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

def sync_configs(drive_path: Path, ui_components: Dict[str, Any], silent: bool = False):
    """
    Sinkronisasi konfigurasi antara lokal dan Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        silent: Jika True, tidak menampilkan output UI
    """
    try:
        from smartcash.common.config_sync import sync_all_configs
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator('info', f'{ICONS["processing"]} Sinkronisasi Konfigurasi'))
        
        # Sinkronisasi semua file konfigurasi
        results = sync_all_configs(
            drive_configs_dir=str(drive_path / 'configs'),
            sync_strategy='drive_priority'
        )
        
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                success_count = len(results.get("success", []))
                failure_count = len(results.get("failure", []))
                
                if failure_count == 0:
                    display(create_status_indicator('success', f"Sinkronisasi berhasil: {success_count} file ✓"))
                else:
                    display(create_status_indicator('warning', f"Sinkronisasi selesai dengan peringatan: {success_count} berhasil, {failure_count} gagal"))
    except ImportError:
        if not silent and 'status' in ui_components:
            with ui_components['status']:
                display(create_status_indicator('warning', f"Modul config_sync tidak tersedia, sinkronisasi manual diperlukan"))