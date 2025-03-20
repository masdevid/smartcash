"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan integrasi UI utils
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.utils.drive_utils import detect_drive_mount

def handle_drive_connection(ui_components: Dict[str, Any]):
    """
    Hubungkan ke Google Drive dan setup struktur proyek dengan integrasi UI utils.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        clear_output()
        display(create_info_alert("Menghubungkan ke Google Drive...", "info", ICONS['processing']))
    
    try:
        # Mount drive dan dapatkan path
        drive_path = mount_google_drive(ui_components)
        if not drive_path:
            return
        
        # Update status panel dengan create_info_alert
        with ui_components['status']:
            clear_output()
            display(create_info_alert("Google Drive berhasil terhubung!", "success", ICONS['success']))
            
            # Buat symlinks
            try:
                create_symlinks(drive_path, ui_components)
            except Exception as e:
                display(create_info_alert(f"Error saat membuat symlinks: {str(e)}", "warning", ICONS['warning']))
            
            # Sinkronisasi konfigurasi
            try:
                sync_configs(drive_path, ui_components)
            except Exception as e:
                display(create_info_alert(f"Error saat sinkronisasi konfigurasi: {str(e)}", "warning", ICONS['warning']))
        
        # Update panel Colab menggunakan create_info_alert
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
        with ui_components['status']:
            clear_output()
            display(create_info_alert(f"Error saat menghubungkan ke Google Drive: {str(e)}", "error", ICONS['error']))
        
        # Log error jika logger tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].error(f"❌ Error koneksi Google Drive: {str(e)}")

def mount_google_drive(ui_components: Dict[str, Any]) -> Optional[Path]:
    """
    Mount Google Drive jika belum ter-mount dengan integrasi drive_utils.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Path direktori SmartCash di Google Drive atau None jika gagal
    """
    try:
        # Gunakan utility dari drive_utils
        is_mounted, drive_path = detect_drive_mount()
        
        if not is_mounted:
            from google.colab import drive
            drive.mount('/content/drive')
            is_mounted, drive_path = detect_drive_mount()
            
            if not is_mounted:
                with ui_components['status']:
                    display(create_info_alert("Gagal mount Google Drive", "error", ICONS['error']))
                return None
        
        # Buat direktori SmartCash di Drive jika belum ada
        smartcash_dir = Path(drive_path) / 'SmartCash'
        os.makedirs(smartcash_dir, exist_ok=True)
        
        return smartcash_dir
    except Exception as e:
        with ui_components['status']:
            display(create_info_alert(f"Error saat mounting Google Drive: {str(e)}", "error", ICONS['error']))
        return None

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any]):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
    """
    # Mapping direktori yang akan dibuat symlink
    symlinks = {
        'data': drive_path / 'data',
        'configs': drive_path / 'configs',
        'runs': drive_path / 'runs',
        'logs': drive_path / 'logs',
        'checkpoints': drive_path / 'checkpoints'
    }
    
    with ui_components['status']:
        display(HTML(f"""
            <div style="margin-top:10px;">
                <h3 style="color:{COLORS['secondary']}; margin:5px 0">🔗 Membuat Symlinks</h3>
            </div>
        """))
    
    for local_name, target_path in symlinks.items():
        with ui_components['status']:
            # Pastikan direktori target ada
            target_path.mkdir(parents=True, exist_ok=True)
            
            local_path = Path(local_name)
            
            # Hapus direktori lokal jika sudah ada
            if local_path.exists() and not local_path.is_symlink():
                backup_path = local_path.with_name(f"{local_name}_backup")
                display(create_status_indicator('info', f"Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup"))
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                local_path.rename(backup_path)
            
            # Buat symlink jika belum ada
            if not local_path.exists():
                local_path.symlink_to(target_path)
                display(create_status_indicator('success', f"Symlink dibuat: {local_name} → {target_path}"))

def sync_configs(drive_path: Path, ui_components: Dict[str, Any]):
    """
    Sinkronisasi konfigurasi antara lokal dan Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
    """
    # Pastikan direktori configs ada
    local_configs = Path('configs')
    drive_configs = drive_path / 'configs'
    
    local_configs.mkdir(parents=True, exist_ok=True)
    drive_configs.mkdir(parents=True, exist_ok=True)
    
    with ui_components['status']:
        display(HTML(f"""
            <div style="margin-top:10px;">
                <h3 style="color:{COLORS['secondary']}; margin:5px 0">{ICONS['processing']} Sinkronisasi Konfigurasi</h3>
            </div>
        """))
    
    # Cek file YAML di lokal dan drive
    local_yamls = list(local_configs.glob('*.yaml')) + list(local_configs.glob('*.yml'))
    drive_yamls = list(drive_configs.glob('*.yaml')) + list(drive_configs.glob('*.yml'))
    
    # Mapping by filename
    local_map = {f.name: f for f in local_yamls}
    drive_map = {f.name: f for f in drive_yamls}
    
    all_files = set(local_map.keys()) | set(drive_map.keys())
    
    for filename in all_files:
        local_file = local_map.get(filename)
        drive_file = drive_map.get(filename)
        
        with ui_components['status']:
            # Hanya file lokal ada
            if local_file and filename not in drive_map:
                shutil.copy2(local_file, drive_configs / filename)
                display(create_status_indicator('info', f"⬆️ File lokal disalin ke Drive: {filename}"))
            
            # Hanya file drive ada
            elif drive_file and filename not in local_map:
                shutil.copy2(drive_file, local_configs / filename)
                display(create_status_indicator('info', f"⬇️ File Drive disalin ke lokal: {filename}"))
            
            # Kedua file ada, bandingkan timestamp
            elif local_file and drive_file:
                # Handle the SameFileError case
                try:
                    if os.path.samefile(local_file, drive_file):
                        display(create_status_indicator('info', f"ℹ️ File sudah sinkron (symlink): {filename}"))
                        continue
                except OSError:
                    pass
                    
                try:
                    local_time = local_file.stat().st_mtime
                    drive_time = drive_file.stat().st_mtime
                    
                    if local_time > drive_time:
                        shutil.copy2(local_file, drive_file)
                        display(create_status_indicator('info', f"⬆️ File lokal lebih baru, disalin ke Drive: {filename}"))
                    else:
                        shutil.copy2(drive_file, local_file)
                        display(create_status_indicator('info', f"⬇️ File Drive lebih baru, disalin ke lokal: {filename}"))
                except shutil.SameFileError:
                    display(create_status_indicator('info', f"ℹ️ File sudah sinkron (symlink): {filename}"))
                except Exception as e:
                    display(create_status_indicator('warning', f"Error saat sinkronisasi {filename}: {str(e)}"))