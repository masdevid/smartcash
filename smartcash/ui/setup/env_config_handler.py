"""
File: smartcash/ui/01_setup/env_config_handler.py
Deskripsi: Handler untuk konfigurasi environment SmartCash, mengelola koneksi Drive dan setup direktori
"""

import os
import shutil
from pathlib import Path
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional, List, Tuple
import yaml

from smartcash.ui.handlers.environment_handler import detect_environment, filter_drive_tree, fallback_get_directory_tree, sync_configs
from smartcash.ui.handlers.error_handler import handle_error, setup_error_handlers
from smartcash.ui.handlers.observer_handler import setup_observer_handlers
from smartcash.ui.handlers.config_handler import setup_config_handlers, update_config
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_helpers import run_async_task, create_progress_updater

def setup_env_config_handlers(ui_components: Dict[str, Any], env: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk komponen env_config.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah ditambahkan handler
    """
    # Setup basic handlers
    ui_components = setup_error_handlers(ui_components)
    ui_components = setup_observer_handlers(ui_components, "env_config")
    ui_components = setup_config_handlers(ui_components, config)
    
    # Setup logger
    logger = setup_ipython_logging(ui_components, "env_config")
    if logger:
        ui_components['logger'] = logger
    
    # Detect environment jika ada env manager
    is_colab = detect_environment(ui_components, env)
    
    # Set progress updater
    progress_updater = create_progress_updater(ui_components['progress_bar'])
    ui_components['progress_updater'] = progress_updater
    
    # Register event handlers
    ui_components['drive_button'].on_click(lambda b: handle_drive_connection(ui_components))
    ui_components['dir_button'].on_click(lambda b: handle_dir_setup(ui_components))
    
    return ui_components

def handle_drive_connection(ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol koneksi Drive.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    status = ui_components['status']
    dir_output = ui_components['dir_output']
    progress_updater = ui_components['progress_updater']
    logger = ui_components.get('logger')
    
    # Clear outputs
    with status:
        clear_output(wait=True)
    with dir_output:
        clear_output(wait=True)
    
    # Update progress
    progress_updater(0, 4, "Memulai koneksi Google Drive")
    
    def mount_drive_task():
        try:
            env = ui_components.get('env')
            
            # Mount Drive dengan env_manager atau fallback
            if env and hasattr(env, 'mount_drive'):
                progress_updater(1, 4, "Mounting Google Drive")
                drive_path = env.mount_drive()
                
                # Create symlinks jika didukung
                if hasattr(env, 'create_symlinks'):
                    progress_updater(2, 4, "Membuat symlinks")
                    env.create_symlinks()
            else:
                # Fallback untuk mounting drive
                progress_updater(1, 4, "Mounting Google Drive (mode fallback)")
                from google.colab import drive
                drive.mount('/content/drive')
                drive_path = '/content/drive/MyDrive/SmartCash'
                
                # Ensure SmartCash directory exists
                os.makedirs(drive_path, exist_ok=True)
            
            # Sync configs
            progress_updater(3, 4, "Sinkronisasi konfigurasi")
            source_dirs = [Path('configs'), Path('/content/configs')]
            target_dirs = [Path(drive_path) / 'configs']
            
            if logger:
                logger.info(f"🔄 Sinkronisasi config antara {source_dirs} → {target_dirs}")
            
            sync_configs(source_dirs, target_dirs, logger)
            
            # Generate dan tampilkan tree
            progress_updater(4, 4, "Mempersiapkan tampilan direktori")
            
            if env and hasattr(env, 'get_directory_tree'):
                tree_html = env.get_directory_tree('/content/drive', max_depth=3)
                tree_html = filter_drive_tree(tree_html)
            else:
                tree_html = fallback_get_directory_tree('/content/drive', max_depth=3)
            
            with dir_output:
                clear_output(wait=True)
                display(HTML(tree_html))
            
            # Update status
            with status:
                display(HTML(f"""
                <div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                    <h3 style="margin-top:0">✅ Google Drive Terhubung</h3>
                    <p>Drive berhasil terhubung dan dikonfigurasi. Struktur direktori SmartCash telah ditampilkan.</p>
                </div>
                """))
            
            if logger:
                logger.success(f"✅ Google Drive berhasil terhubung ke {drive_path}")
            
            # Update config
            update_config(ui_components, {
                'environment': {
                    'drive_mounted': True,
                    'drive_path': drive_path
                }
            })
            
            return True
            
        except Exception as e:
            handle_error(e, status)
            if logger:
                logger.error(f"❌ Gagal menghubungkan Google Drive: {str(e)}")
            progress_updater(0, 4, "Gagal: Koneksi Google Drive")
            return False
    
    # Run task async
    run_async_task(mount_drive_task)

def handle_dir_setup(ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol setup direktori.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    status = ui_components['status']
    dir_output = ui_components['dir_output']
    progress_updater = ui_components['progress_updater']
    logger = ui_components.get('logger')
    
    # Clear outputs
    with status:
        clear_output(wait=True)
    with dir_output:
        clear_output(wait=True)
    
    # Update progress
    progress_updater(0, 3, "Memulai setup direktori")
    
    def setup_directory_task():
        try:
            env = ui_components.get('env')
            config = ui_components.get('config', {})
            
            # Get project dirs from config
            data_dir = config.get('data', {}).get('dir', 'data')
            processed_dir = config.get('data', {}).get('processed_dir', 'data/preprocessed')
            
            # Subdirectories to create
            subdirs = [
                'data/train', 'data/valid', 'data/test',
                'data/preprocessed/train', 'data/preprocessed/valid', 'data/preprocessed/test',
                'configs', 'logs', 'runs', 'visualizations'
            ]
            
            progress_updater(1, 3, "Membuat struktur direktori")
            
            # Ensure project directories exist using env_manager or fallback
            if env and hasattr(env, 'setup_project_structure'):
                env.setup_project_structure()
            else:
                # Fallback: create directories manually
                for subdir in subdirs:
                    os.makedirs(subdir, exist_ok=True)
            
            progress_updater(2, 3, "Mempersiapkan tampilan direktori")
            
            # Generate dan tampilkan tree
            if env and hasattr(env, 'get_directory_tree'):
                tree_html = env.get_directory_tree('.', max_depth=3)
            else:
                tree_html = fallback_get_directory_tree('.', max_depth=3)
            
            with dir_output:
                clear_output(wait=True)
                display(HTML(tree_html))
            
            # Update status
            progress_updater(3, 3, "Setup direktori selesai")
            with status:
                display(HTML(f"""
                <div style="padding:10px;background:#d4edda;border-left:4px solid #155724;color:#155724;margin:10px 0">
                    <h3 style="margin-top:0">✅ Direktori Berhasil Dibuat</h3>
                    <p>Struktur direktori SmartCash telah dibuat dan siap digunakan.</p>
                </div>
                """))
            
            if logger:
                logger.success(f"✅ Struktur direktori proyek berhasil dibuat")
            
            return True
            
        except Exception as e:
            handle_error(e, status)
            if logger:
                logger.error(f"❌ Gagal membuat struktur direktori: {str(e)}")
            progress_updater(0, 3, "Gagal: Setup direktori")
            return False
    
    # Run task async
    run_async_task(setup_directory_task)