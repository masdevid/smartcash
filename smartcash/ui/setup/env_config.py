"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan perbaikan inisialisasi dan error handling
"""

from typing import Dict, Any
from IPython.display import display

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi fallback_utils."""
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
        
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger: 
            ui_components['logger'] = logger
            logger.info("✅ Logger environment config berhasil diinisialisasi")
        
        # Inisialisasi default konfigurasi jika diperlukan
        try:
            verify_default_configs(ui_components)
        except Exception as e:
            if logger: logger.warning(f"⚠️ Error saat verifikasi konfigurasi: {str(e)}")
        
        # Setup Drive sync initializer
        try:
            from smartcash.ui.setup.drive_sync_initializer import initialize_drive_sync
            initialize_drive_sync(ui_components)
        except ImportError as e:
            if logger: logger.debug(f"ℹ️ Drive sync initializer tidak tersedia: {str(e)}")
        
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
    """Verifikasi dan pastikan konfigurasi default tersedia."""
    logger = ui_components.get('logger')
    
    try:
        # Import required modules
        import os
        import yaml
        from pathlib import Path
        
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
        
        # Cek dan buat template konfigurasi jika diperlukan
        created_files = []
        for config_file in required_configs:
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
                
                created_files.append(config_file)
        
        if created_files and logger:
            logger.info(f"✅ Membuat {len(created_files)} file konfigurasi default: {', '.join(created_files)}")
        
        # Coba sinkronisasi konfigurasi jika tersedia
        try:
            from smartcash.common.config_sync import sync_all_configs
            result = sync_all_configs(create_backup=True)
            if logger:
                logger.info(f"✅ Sinkronisasi konfigurasi: {len(result.get('success', []))} berhasil, {len(result.get('failure', []))} gagal")
        except ImportError:
            pass
            
    except Exception as e:
        # Log error tapi jangan gagalkan inisialisasi
        if logger:
            logger.warning(f"⚠️ Error verifikasi konfigurasi: {str(e)}")