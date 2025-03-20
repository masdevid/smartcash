"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan integrasi sinkronisasi Drive
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional

def initialize_drive_sync():
    """Inisialisasi dan sinkronisasi Google Drive sebelum setup environment"""
    try:
        # Coba gunakan colab_initializer jika tersedia
        from smartcash.ui.setup.colab_initializer import initialize_environment
        initialize_environment()
    except ImportError:
        # Fallback minimal: mount drive & buat direktori
        import os, sys
        
        # Install package dasar jika perlu
        try:
            import yaml, tqdm, ipywidgets
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets", "tqdm", "pyyaml"])
        
        # Mount Drive jika di Google Colab
        if 'google.colab' in sys.modules and not os.path.exists('/content/drive/MyDrive'):
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Buat direktori dasar di Drive jika belum ada
            drive_dir = '/content/drive/MyDrive/SmartCash'
            if not os.path.exists(drive_dir):
                os.makedirs(f"{drive_dir}/configs", exist_ok=True)
                os.makedirs(f"{drive_dir}/data", exist_ok=True)
                os.makedirs(f"{drive_dir}/runs", exist_ok=True)
                os.makedirs(f"{drive_dir}/logs", exist_ok=True)

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi utilities"""
    
    # Import komponen dengan pendekatan konsolidasi
    from smartcash.ui.setup.env_config_component import create_env_config_ui
    from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component
    from smartcash.ui.utils.logging_utils import setup_ipython_logging
    
    try:
        # Setup notebook environment
        env, config = setup_notebook_environment("env_config")
        
        # Buat komponen UI dengan helpers
        ui_components = create_env_config_ui(env, config)
        
        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Modul environment config berhasil dimuat")
        
        # Setup handlers untuk UI
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Cek fungsionalitas drive_handler yang juga berhubungan dengan sinkronisasi konfigurasi
        try:
            from smartcash.ui.setup.drive_handler import setup_drive_handler, handle_drive_connection
            ui_components = setup_drive_handler(ui_components, env, config, auto_connect=True)
        except ImportError as e:
            if logger:
                logger.debug(f"Module drive_handler tidak tersedia: {str(e)}")
                
        # Cek juga fungsionalitas config_sync
        try:
            from smartcash.common.config_sync import sync_all_configs
            from smartcash.common.environment import get_environment_manager
            
            env_manager = get_environment_manager()
            if env_manager.is_drive_mounted:
                if logger:
                    logger.info("🔄 Sinkronisasi konfigurasi dengan Google Drive...")
                results = sync_all_configs(sync_strategy='drive_priority')
                
                success_count = len(results.get("success", []))
                failure_count = len(results.get("failure", []))
                if logger:
                    logger.info(f"✅ Sinkronisasi selesai: {success_count} berhasil, {failure_count} gagal")
        except ImportError:
            if logger:
                logger.debug("Module config_sync tidak tersedia")
        
    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import import_with_fallback, show_status
        
        # Fallback environment setup
        env = type('DummyEnv', (), {
            'is_colab': 'google.colab' in __import__('sys').modules,
            'base_dir': __import__('os').getcwd(),
            'is_drive_mounted': False,
        })
        config = {}
        
        # Buat UI components
        ui_components = create_env_config_ui(env, config)
        
        # Tampilkan pesan error
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components