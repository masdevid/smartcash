"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dan pembuatan symlinks dengan UI yang lebih baik
"""

import os
import sys
from pathlib import Path
import shutil
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS

def handle_drive_connection(ui_components: Dict[str, Any]):
    """
    Hubungkan ke Google Drive dan setup struktur proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        clear_output()
        display(HTML(f"""
            <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                      color:{COLORS['alert_info_text']}; 
                      border-radius:4px; margin:5px 0;
                      border-left:4px solid {COLORS['alert_info_text']};">
                <p style="margin:5px 0">{ICONS['processing']} Menghubungkan ke Google Drive...</p>
            </div>
        """))
    
    try:
        # Mount drive dan dapatkan path
        drive_path = mount_google_drive(ui_components)
        if not drive_path:
            return
        
        # Update status panel
        with ui_components['status']:
            clear_output()
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                          color:{COLORS['alert_success_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_success_text']};">
                    <p style="margin:5px 0">{ICONS['success']} Google Drive berhasil terhubung!</p>
                </div>
            """))
            
            # Buat symlinks
            try:
                create_symlinks(drive_path, ui_components)
            except Exception as e:
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                              color:{COLORS['alert_warning_text']}; 
                              border-radius:4px; margin:5px 0;
                              border-left:4px solid {COLORS['alert_warning_text']};">
                        <p style="margin:5px 0">{ICONS['warning']} Error saat membuat symlinks: {str(e)}</p>
                    </div>
                """))
            
            # Sinkronisasi konfigurasi
            try:
                sync_configs(drive_path, ui_components)
            except Exception as e:
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                              color:{COLORS['alert_warning_text']}; 
                              border-radius:4px; margin:5px 0;
                              border-left:4px solid {COLORS['alert_warning_text']};">
                        <p style="margin:5px 0">{ICONS['warning']} Error saat sinkronisasi konfigurasi: {str(e)}</p>
                    </div>
                """))
        
        # Update panel Colab
        ui_components['colab_panel'].value = f"""
        <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                  color:{COLORS['alert_success_text']}; 
                  border-radius:4px; margin:10px 0;
                  border-left:4px solid {COLORS['alert_success_text']};">
            <h3 style="color:inherit; margin:5px 0">🔍 Environment: Google Colab</h3>
            <p style="margin:5px 0">{ICONS['success']} Status Google Drive: <strong>terhubung</strong></p>
            <p style="margin:5px 0">Drive terhubung dan struktur direktori telah dibuat.</p>
        </div>
        """
    except Exception as e:
        with ui_components['status']:
            clear_output()
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                          color:{COLORS['alert_danger_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_danger_text']};">
                    <p style="margin:5px 0">{ICONS['error']} Error saat menghubungkan ke Google Drive: {str(e)}</p>
                </div>
            """))

def mount_google_drive(ui_components: Dict[str, Any]) -> Optional[Path]:
    """
    Mount Google Drive jika belum ter-mount.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Path direktori SmartCash di Google Drive atau None jika gagal
    """
    try:
        from google.colab import drive
        
        # Cek apakah drive sudah ter-mount
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        
        # Buat direktori SmartCash di Drive jika belum ada
        drive_path = Path('/content/drive/MyDrive/SmartCash')
        os.makedirs(drive_path, exist_ok=True)
        
        return drive_path
    except Exception as e:
        with ui_components['status']:
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                          color:{COLORS['alert_danger_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_danger_text']};">
                    <p style="margin:5px 0">{ICONS['error']} Error saat mounting Google Drive: {str(e)}</p>
                </div>
            """))
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
            <div style="margin-top:10px">
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
                display(HTML(f"""
                    <div style="padding:4px; color:{COLORS['alert_info_text']};">
                        <p style="margin:5px 0">{ICONS['processing']} Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup</p>
                    </div>
                """))
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                local_path.rename(backup_path)
            
            # Buat symlink jika belum ada
            if not local_path.exists():
                local_path.symlink_to(target_path)
                display(HTML(f"""
                    <div style="padding:4px; color:{COLORS['alert_success_text']};">
                        <p style="margin:5px 0">{ICONS['success']} Symlink dibuat: {local_name} → {target_path}</p>
                    </div>
                """))

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
            <div style="margin-top:10px">
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
                display(HTML(f"""
                    <div style="padding:4px; color:{COLORS['alert_info_text']};">
                        <p style="margin:5px 0">⬆️ File lokal disalin ke Drive: {filename}</p>
                    </div>
                """))
            
            # Hanya file drive ada
            elif drive_file and filename not in local_map:
                shutil.copy2(drive_file, local_configs / filename)
                display(HTML(f"""
                    <div style="padding:4px; color:{COLORS['alert_info_text']};">
                        <p style="margin:5px 0">⬇️ File Drive disalin ke lokal: {filename}</p>
                    </div>
                """))
            
            # Kedua file ada, bandingkan timestamp
            elif local_file and drive_file:
                # Handle the SameFileError case - check if the files are the same
                try:
                    if os.path.samefile(local_file, drive_file):
                        display(HTML(f"""
                            <div style="padding:4px; color:{COLORS['alert_info_text']};">
                                <p style="margin:5px 0">ℹ️ File sudah sinkron (symlink): {filename}</p>
                            </div>
                        """))
                        continue
                except OSError:
                    # On some systems samefile may raise OSError
                    pass
                    
                try:
                    local_time = local_file.stat().st_mtime
                    drive_time = drive_file.stat().st_mtime
                    
                    if local_time > drive_time:
                        shutil.copy2(local_file, drive_file)
                        display(HTML(f"""
                            <div style="padding:4px; color:{COLORS['alert_info_text']};">
                                <p style="margin:5px 0">⬆️ File lokal lebih baru, disalin ke Drive: {filename}</p>
                            </div>
                        """))
                    else:
                        shutil.copy2(drive_file, local_file)
                        display(HTML(f"""
                            <div style="padding:4px; color:{COLORS['alert_info_text']};">
                                <p style="margin:5px 0">⬇️ File Drive lebih baru, disalin ke lokal: {filename}</p>
                            </div>
                        """))
                except shutil.SameFileError:
                    display(HTML(f"""
                        <div style="padding:4px; color:{COLORS['alert_info_text']};">
                            <p style="margin:5px 0">ℹ️ File sudah sinkron (symlink): {filename}</p>
                        </div>
                    """))
                except Exception as e:
                    display(HTML(f"""
                        <div style="padding:4px; color:{COLORS['alert_warning_text']};">
                            <p style="margin:5px 0">{ICONS['warning']} Error saat sinkronisasi {filename}: {str(e)}</p>
                        </div>
                    """))