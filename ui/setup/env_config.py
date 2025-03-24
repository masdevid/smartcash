"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan integrasi UI logger
"""

from typing import Dict, Any
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi UI logger."""
    # Inisialisasi ui_components dengan nilai default
    ui_components = {'status': None, 'module_name': 'env_config'}
    
    try:
        # Import modul dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, log_to_ui
        
        # Setup environment dan komponen UI
        env, config = setup_notebook_environment("env_config")
        ui_components = create_env_config_ui(env, config)
        
        # Setup logging dan log inisialisasi
        if 'status' in ui_components: 
            log_to_ui(ui_components, "🚀 Inisialisasi environment config dimulai", "info")
        
        # Setup logger dengan arahkan ke UI
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger: 
            ui_components['logger'] = logger
            logger.info("✅ Logger environment config berhasil diinisialisasi")
        
        # Inisialisasi default konfigurasi menggunakan ThreadPoolExecutor
        def init_configs_async():
            """Initialize configs in background thread"""
            try:
                # Jalankan verifikasi konfigurasi
                verify_default_configs(ui_components)
                
                # Setup Drive sync initializer
                try:
                    from smartcash.common.drive_sync_initializer import initialize_configs
                    success, message = initialize_configs(logger)
                    if logger: logger.info(f"🔄 Sinkronisasi konfigurasi: {message}")
                except ImportError as e:
                    if logger: logger.debug(f"ℹ️ Drive sync initializer tidak tersedia: {str(e)}")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat inisialisasi konfigurasi: {str(e)}")
        
        # Jalankan inisialisasi konfigurasi di thread terpisah agar tidak blokir UI
        with ThreadPoolExecutor(max_workers=1) as executor:
            config_future = executor.submit(init_configs_async)
        
        # Setup handlers
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Log selesai inisialisasi
        if logger: logger.info("✅ Environment config selesai diinisialisasi")
    
    except Exception as e:
        # Fallback sederhana jika terjadi error
        try:
            from smartcash.ui.utils.fallback_utils import create_fallback_ui, show_status
            ui_components = create_fallback_ui(ui_components, f"❌ Error saat inisialisasi environment config: {str(e)}", "error")
            show_status(f"Error: {str(e)}", "error", ui_components)
            
            # Log error jika logger tersedia
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Error setup environment config: {str(e)}")
        except ImportError:
            # Jika fallback_utils tidak tersedia, gunakan widgets standar
            import ipywidgets as widgets
            from IPython.display import display, HTML
            
            # Buat fallback UI sangat sederhana
            header = widgets.HTML("<h3>⚙️ Environment Config</h3>")
            error_msg = widgets.HTML(f"<div style='color:red;padding:10px;border:1px solid red;'>❌ Error: {str(e)}</div>")
            ui_components['ui'] = widgets.VBox([header, error_msg])
            
            # Display error
            print(f"Error initializing environment config: {str(e)}")
    
    # Return ui_components
    return ui_components

def verify_default_configs(ui_components: Dict[str, Any]) -> None:
    """Verifikasi dan pastikan konfigurasi default tersedia dengan one-liner dan smart file handling."""
    logger = ui_components.get('logger')
    
    try:
        # Import required modules
        import os, yaml, json, shutil
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor
        
        # Pastikan direktori configs ada
        config_dir = Path('configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # List konfigurasi default yang diperlukan
        required_configs = [
            "base_config.yaml",
            "colab_config.yaml",
            "dataset_config.yaml",
            "preprocessing_config.yaml",
            "training_config.yaml",
            "augmentation_config.yaml",
            "evaluation_config.yaml",
            "model_config.yaml"
        ]
        
        # Cek dan buat template konfigurasi jika diperlukan secara paralel
        def process_config_file(config_file):
            """Process satu file konfigurasi"""
            config_path = config_dir / config_file
            if not config_path.exists():
                # Buat template sederhana
                basic_config = {
                    '_base_': 'base_config.yaml' if config_file != "base_config.yaml" else None,
                    'created_date': 'auto-generated',
                    'config_type': config_file.replace('_config.yaml', '')
                }
                
                # Simpan konfigurasi dasar
                with open(config_path, 'w') as f:
                    yaml.dump(basic_config, f, default_flow_style=False)
                
                return config_file  # Return nama file yang berhasil dibuat
            return None
        
        # Paralel processing untuk pembuatan file konfigurasi
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_config_file, required_configs))
        
        # Filter None dan dapatkan daftar created_files
        created_files = [f for f in results if f]
        
        if created_files and logger:
            logger.info(f"✅ Membuat {len(created_files)} file konfigurasi default: {', '.join(created_files)}")
        
        # Coba sinkronisasi konfigurasi jika tersedia (menggunakan modul baru)
        try:
            from smartcash.common.config_sync import sync_all_configs
            result = sync_all_configs(sync_strategy='merge', create_backup=True)
            if logger:
                logger.info(f"✅ Sinkronisasi konfigurasi: {len(result.get('success', []))} berhasil, {len(result.get('failure', []))} gagal")
        except ImportError:
            pass
            
    except Exception as e:
        # Log error tapi jangan gagalkan inisialisasi
        if logger:
            logger.warning(f"⚠️ Error verifikasi konfigurasi: {str(e)}")