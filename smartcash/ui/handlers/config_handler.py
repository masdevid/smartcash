"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk konfigurasi environment SmartCash dengan deteksi dan status komprehensif
"""

from typing import Dict, Any, Optional
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import platform
import sys

def setup_env_config_handlers(
    ui_components: Dict[str, Any], 
    env: Optional[Any] = None, 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi environment dengan status komprehensif.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi environment
        
    Returns:
        Dictionary UI components yang telah ditambahkan handler
    """
    # Import dependencies
    from smartcash.ui.handlers.environment_handler import (
        detect_environment, 
        check_smartcash_dir, 
        filter_drive_tree,
        sync_configs
    )
    from smartcash.ui.handlers.config_handler import setup_config_handlers
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    
    # Setup config dan observer handlers
    ui_components = setup_config_handlers(ui_components, config)
    ui_components = setup_observer_handlers(ui_components)
    
    # Logger
    logger = ui_components.get('logger')
    
    def update_environment_status():
        """Update status lingkungan secara komprehensif."""
        try:
            # Deteksi lingkungan
            is_colab = detect_environment(ui_components, env)
            
            # Informasi sistem dasar
            system_info = {
                'Environment': 'Google Colab' if is_colab else 'Lokal',
                'Python Version': platform.python_version(),
                'Platform': f"{platform.system()} {platform.release()}",
                'Base Directory': str(Path.cwd())
            }
            
            # Deteksi CUDA/GPU
            try:
                import torch
                system_info.update({
                    'CUDA Available': torch.cuda.is_available(),
                    'GPU Device Count': torch.cuda.device_count(),
                    'Current GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak Ada'
                })
            except ImportError:
                system_info.update({
                    'CUDA Available': 'Tidak Dapat Dideteksi',
                    'GPU Device Count': 0,
                    'Current GPU': 'Tidak Dapat Dideteksi'
                })
            
            # Deteksi Google Drive
            drive_status = 'Tidak Ter-mount'
            drive_path = 'Tidak Tersedia'
            
            if is_colab:
                try:
                    from google.colab import drive
                    from pathlib import Path
                    
                    drive_mount_path = Path('/content/drive/MyDrive')
                    if drive_mount_path.exists():
                        drive_status = 'Ter-mount'
                        drive_path = str(drive_mount_path)
                except ImportError:
                    pass
            
            system_info['Google Drive'] = {
                'Status': drive_status,
                'Path': drive_path
            }
            
            # Tampilkan status di info panel
            with ui_components['info_panel']:
                clear_output(wait=True)
                status_html = "<div style='background:#f8f9fa; padding:15px; border-radius:5px;'>"
                status_html += "<h4>📊 Informasi Lingkungan Sistem</h4>"
                status_html += "<ul>"
                for key, value in system_info.items():
                    if isinstance(value, dict):
                        status_html += f"<li><strong>{key}:</strong>"
                        status_html += "<ul>"
                        for subkey, subvalue in value.items():
                            status_html += f"<li>{subkey}: {subvalue}</li>"
                        status_html += "</ul></li>"
                    else:
                        status_html += f"<li><strong>{key}:</strong> {value}</li>"
                status_html += "</ul></div>"
                
                display(HTML(status_html))
            
            return system_info
        except Exception as e:
            if logger:
                logger.error(f"❌ Error updating environment status: {str(e)}")
            return {}
    
    def handle_drive_connection(b):
        """Handler untuk koneksi Google Drive."""
        try:
            # Pastikan environment manager tersedia
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager(logger=logger)
            
            # Mount drive
            mount_success, mount_msg = env_manager.mount_drive()
            
            if mount_success:
                # Buat symlink
                symlink_stats = env_manager.create_symlinks()
                
                # Sinkronisasi konfigurasi
                sync_configs(
                    [env_manager.get_path('configs')],
                    [env_manager.drive_path / 'configs'],
                    logger
                )
                
                # Update status
                update_environment_status()
                
                # Tampilkan tree drive
                with ui_components['info_panel']:
                    tree_html = env_manager.get_directory_tree(
                        env_manager.drive_path, 
                        max_depth=3
                    )
                    display(HTML(filter_drive_tree(tree_html)))
                
                if logger:
                    logger.success(f"🔗 Google Drive berhasil ter-mount: {mount_msg}")
            else:
                # Tampilkan pesan error
                with ui_components['status']:
                    display(HTML(f"""
                    <div style='color:red; padding:10px; background:#f8d7da; border-radius:4px;'>
                        ❌ Gagal koneksi Google Drive: {mount_msg}
                    </div>
                    """))
        except Exception as e:
            if logger:
                logger.error(f"❌ Error koneksi Drive: {str(e)}")
    
    def handle_local_dir_setup(b):
        """Handler untuk setup direktori lokal."""
        try:
            # Pastikan environment manager tersedia
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager(logger=logger)
            
            # Setup struktur proyek
            dir_stats = env_manager.setup_project_structure()
            
            # Update status
            update_environment_status()
            
            # Tampilkan tree direktori
            with ui_components['info_panel']:
                tree_html = env_manager.get_directory_tree(
                    env_manager.base_dir, 
                    max_depth=3
                )
                display(HTML(tree_html))
                
            if logger:
                logger.success(f"📁 Setup direktori berhasil: {dir_stats}")
        except Exception as e:
            if logger:
                logger.error(f"❌ Error setup direktori: {str(e)}")
    
    # Tambahkan handler ke tombol
    ui_components['drive_button'].on_click(handle_drive_connection)
    ui_components['dir_button'].on_click(handle_local_dir_setup)
    
    # Deteksi environment awal
    update_environment_status()
    
    # Cek direktori smartcash
    check_smartcash_dir(ui_components)
    
    return ui_components