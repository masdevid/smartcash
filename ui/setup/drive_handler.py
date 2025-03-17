"""
File: smartcash/ui/setup/drive_handler.py
Deskripsi: Handler untuk koneksi Google Drive dengan komponen UI yang terintegrasi
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

def handle_drive_connection(ui_components: Dict[str, Any]):
    """
    Hubungkan ke Google Drive dan setup struktur proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    from smartcash.ui.utils.ui_helpers import update_output_area
    from smartcash.ui.handlers.error_handler import handle_error
    from smartcash.ui.utils.constants import ICONS

    # Clear output dan tampilkan status awal
    with ui_components['status']:
        clear_output()
    
    update_output_area(
        ui_components['status'], 
        f"{ICONS['processing']} Menghubungkan ke Google Drive...", 
        'info'
    )
    
    try:
        # Coba dapatkan logger
        logger = ui_components.get('logger', None)
        
        # Mount drive dan dapatkan path
        drive_path = mount_google_drive(ui_components, logger)
        if not drive_path:
            return
        
        # Update status panel
        update_output_area(
            ui_components['status'], 
            f"{ICONS['success']} Google Drive berhasil terhubung!", 
            'success', 
            True
        )
            
        # Buat symlinks
        try:
            create_symlinks(drive_path, ui_components, logger)
        except Exception as e:
            update_output_area(
                ui_components['status'],
                f"{ICONS['warning']} Error saat membuat symlinks: {str(e)}",
                'warning'
            )
            if logger:
                logger.warning(f"⚠️ Error saat membuat symlinks: {str(e)}")
            
        # Sinkronisasi konfigurasi
        try:
            sync_configs(drive_path, ui_components, logger)
        except Exception as e:
            update_output_area(
                ui_components['status'],
                f"{ICONS['warning']} Error saat sinkronisasi konfigurasi: {str(e)}",
                'warning'
            )
            if logger:
                logger.warning(f"⚠️ Error saat sinkronisasi konfigurasi: {str(e)}")
        
        # Update panel Colab
        from smartcash.ui.utils.constants import COLORS
        ui_components['colab_panel'].value = f"""
        <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; color:{COLORS['alert_success_text']}; border-radius:4px; margin:10px 0">
            <h3 style="color:{COLORS['alert_success_text']}; margin:5px 0">{ICONS['settings']} Environment: Google Colab</h3>
            <p style="margin:5px 0">{ICONS['success']} Status Google Drive: <strong>terhubung</strong></p>
            <p style="margin:5px 0">Drive terhubung dan struktur direktori telah dibuat.</p>
        </div>
        """
        
        # Tampilkan struktur direktori Drive
        display_drive_tree(drive_path, ui_components)
        
    except Exception as e:
        handle_error(e, ui_components['status'])
        if logger:
            logger.error(f"❌ Error saat menghubungkan ke Google Drive: {str(e)}")

def mount_google_drive(ui_components: Dict[str, Any], logger=None) -> Optional[Path]:
    """
    Mount Google Drive jika belum ter-mount.
    
    Args:
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
        
    Returns:
        Path direktori SmartCash di Google Drive atau None jika gagal
    """
    from smartcash.ui.utils.ui_helpers import update_output_area
    from smartcash.ui.utils.constants import ICONS
    
    try:
        from google.colab import drive
        
        # Cek apakah drive sudah ter-mount
        if not os.path.exists('/content/drive/MyDrive'):
            update_output_area(
                ui_components['status'],
                f"{ICONS['processing']} Mounting Google Drive...",
                'info'
            )
            drive.mount('/content/drive')
        
        # Buat direktori SmartCash di Drive jika belum ada
        drive_path = Path('/content/drive/MyDrive/SmartCash')
        os.makedirs(drive_path, exist_ok=True)
        
        if logger:
            logger.info(f"✅ Google Drive ter-mount pada: {drive_path}")
            
        update_output_area(
            ui_components['status'],
            f"{ICONS['success']} Google Drive ter-mount pada: {drive_path}",
            'success'
        )
        
        return drive_path
    except Exception as e:
        update_output_area(
            ui_components['status'],
            f"{ICONS['error']} Error saat mounting Google Drive: {str(e)}",
            'error'
        )
        if logger:
            logger.error(f"❌ Error saat mounting Google Drive: {str(e)}")
        return None

def create_symlinks(drive_path: Path, ui_components: Dict[str, Any], logger=None):
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
    """
    from smartcash.ui.utils.ui_helpers import update_output_area
    from smartcash.ui.utils.constants import ICONS
    
    # Mapping direktori yang akan dibuat symlink
    symlinks = {
        'data': drive_path / 'data',
        'configs': drive_path / 'configs',
        'runs': drive_path / 'runs',
        'logs': drive_path / 'logs',
        'checkpoints': drive_path / 'checkpoints'
    }
    
    update_output_area(
        ui_components['status'],
        f"{ICONS['processing']} Membuat symlinks...",
        'info'
    )
    
    for local_name, target_path in symlinks.items():
        # Pastikan direktori target ada
        target_path.mkdir(parents=True, exist_ok=True)
        
        local_path = Path(local_name)
        
        # Hapus direktori lokal jika sudah ada
        if local_path.exists() and not local_path.is_symlink():
            backup_path = local_path.with_name(f"{local_name}_backup")
            update_output_area(
                ui_components['status'],
                f"{ICONS['processing']} Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup",
                'info'
            )
            if logger:
                logger.info(f"🔄 Memindahkan direktori lokal ke backup: {local_name} → {local_name}_backup")
                
            if backup_path.exists():
                shutil.rmtree(backup_path)
            local_path.rename(backup_path)
        
        # Buat symlink jika belum ada
        if not local_path.exists():
            local_path.symlink_to(target_path)
            update_output_area(
                ui_components['status'],
                f"{ICONS['success']} Symlink dibuat: {local_name} → {target_path}",
                'success'
            )
            if logger:
                logger.info(f"🔗 Symlink berhasil dibuat: {local_name} → {target_path}")

def sync_configs(drive_path: Path, ui_components: Dict[str, Any], logger=None):
    """
    Sinkronisasi konfigurasi antara lokal dan Google Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
    """
    from smartcash.ui.utils.ui_helpers import update_output_area
    from smartcash.ui.utils.constants import ICONS
    
    # Pastikan direktori configs ada
    local_configs = Path('configs')
    drive_configs = drive_path / 'configs'
    
    local_configs.mkdir(parents=True, exist_ok=True)
    drive_configs.mkdir(parents=True, exist_ok=True)
    
    update_output_area(
        ui_components['status'],
        f"{ICONS['processing']} Sinkronisasi konfigurasi...",
        'info'
    )
    
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
        
        try:
            # Hanya file lokal ada
            if local_file and filename not in drive_map:
                shutil.copy2(local_file, drive_configs / filename)
                update_output_area(
                    ui_components['status'],
                    f"{ICONS['upload']} File lokal disalin ke Drive: {filename}",
                    'info'
                )
                if logger:
                    logger.info(f"⬆️ File lokal disalin ke Drive: {filename}")
            
            # Hanya file drive ada
            elif drive_file and filename not in local_map:
                shutil.copy2(drive_file, local_configs / filename)
                update_output_area(
                    ui_components['status'],
                    f"{ICONS['download']} File Drive disalin ke lokal: {filename}",
                    'info'
                )
                if logger:
                    logger.info(f"⬇️ File Drive disalin ke lokal: {filename}")
            
            # Kedua file ada, bandingkan timestamp
            elif local_file and drive_file:
                # Handle the SameFileError case - check if the files are the same
                if os.path.samefile(local_file, drive_file):
                    update_output_area(
                        ui_components['status'],
                        f"{ICONS['success']} File sudah sinkron (symlink): {filename}",
                        'info'
                    )
                    if logger:
                        logger.info(f"✅ File sudah sinkron (symlink): {filename}")
                    continue
                    
                try:
                    local_time = local_file.stat().st_mtime
                    drive_time = drive_file.stat().st_mtime
                    
                    if local_time > drive_time:
                        shutil.copy2(local_file, drive_file)
                        update_output_area(
                            ui_components['status'],
                            f"{ICONS['upload']} File lokal lebih baru, disalin ke Drive: {filename}",
                            'info'
                        )
                        if logger:
                            logger.info(f"⬆️ File lokal lebih baru, disalin ke Drive: {filename}")
                    else:
                        shutil.copy2(drive_file, local_file)
                        update_output_area(
                            ui_components['status'],
                            f"{ICONS['download']} File Drive lebih baru, disalin ke lokal: {filename}",
                            'info'
                        )
                        if logger:
                            logger.info(f"⬇️ File Drive lebih baru, disalin ke lokal: {filename}")
                except shutil.SameFileError:
                    update_output_area(
                        ui_components['status'],
                        f"{ICONS['success']} File sudah sinkron (symlink): {filename}",
                        'info'
                    )
                    if logger:
                        logger.info(f"✅ File sudah sinkron (symlink): {filename}")
                except Exception as e:
                    update_output_area(
                        ui_components['status'],
                        f"{ICONS['warning']} Error saat sinkronisasi {filename}: {str(e)}",
                        'warning'
                    )
                    if logger:
                        logger.warning(f"⚠️ Error saat sinkronisasi {filename}: {str(e)}")
        except Exception as e:
            update_output_area(
                ui_components['status'],
                f"{ICONS['warning']} Error saat proses file {filename}: {str(e)}",
                'warning'
            )
            if logger:
                logger.warning(f"⚠️ Error saat proses file {filename}: {str(e)}")

def display_drive_tree(drive_path: Path, ui_components: Dict[str, Any]):
    """
    Tampilkan struktur direktori Drive.
    
    Args:
        drive_path: Path direktori di Google Drive
        ui_components: Dictionary komponen UI
    """
    # Tampilkan tree menggunakan fungsi yang sudah ada
    try:
        from smartcash.ui.utils.file_utils import directory_tree
        from smartcash.ui.utils.constants import ICONS
        
        update_output_area(
            ui_components['status'],
            f"{ICONS['folder']} Struktur Direktori Google Drive",
            'info'
        )
        
        # Filter hanya direktori SmartCash
        tree_html = directory_tree(
            drive_path, 
            max_depth=2,
            exclude_patterns=[r"\.git", r"__pycache__", r"\.ipynb_checkpoints"]
        )
        
        with ui_components['status']:
            display(HTML(tree_html))
    except Exception as e:
        # Fallback ke fungsi basic
        tree_html = fallback_get_directory_tree(drive_path)
        with ui_components['status']:
            display(HTML(tree_html))

def fallback_get_directory_tree(root_dir: Path) -> str:
    """
    Fungsi fallback untuk menampilkan struktur direktori jika utility tidak tersedia.
    
    Args:
        root_dir: Path direktori
        
    Returns:
        String HTML dengan tree direktori
    """
    result = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n"
    result += f"<span style='color:#0366d6; font-weight:bold;'>{root_dir.name}/</span>\n"
    
    try:
        # Get top-level directories only
        for item in sorted(root_dir.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                result += f"├─ <span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
                # Include one more level
                try:
                    subpaths = list(item.iterdir())
                    for i, subitem in enumerate(sorted(subpaths)):
                        is_last = i == len(subpaths) - 1
                        connector = "└─ " if is_last else "├─ "
                        
                        if subitem.is_dir():
                            result += f"│  {connector}<span style='color:#0366d6;'>{subitem.name}/</span>\n"
                        else:
                            result += f"│  {connector}{subitem.name}\n"
                except:
                    result += "│  └─ ...\n"
    except Exception as e:
        result += f"<span style='color:red;'>Error reading directory: {str(e)}</span>\n"
        
    result += "</pre>"
    return result

def filter_drive_tree(html_tree: str) -> str:
    """
    Filter struktur direktori Drive untuk hanya menampilkan bagian yang relevan.
    
    Args:
        html_tree: String HTML dengan tree direktori
        
    Returns:
        String HTML yang difilter
    """
    import re
    
    # Pisahkan baris-baris tree
    lines = html_tree.split("\n")
    filtered_lines = []
    
    # Filter untuk direktori SmartCash saja
    in_smartcash = False
    for line in lines:
        if "SmartCash" in line:
            in_smartcash = True
            filtered_lines.append(line)
        elif in_smartcash and ("</pre>" in line or re.match(r".*drive.*My Drive.*", line)):
            in_smartcash = False
            filtered_lines.append(line)
        elif in_smartcash:
            filtered_lines.append(line)
        elif "</pre>" in line:
            filtered_lines.append(line)
    
    return "\n".join(filtered_lines)