"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan pendekatan modular
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

def initialize_env_config_ui():
    """Inisialisasi UI dan handler untuk konfigurasi environment."""
    # Inisialisasi ui_components dengan nilai default
    ui_components = {'status': None, 'module_name': 'env_config'}
    
    try:
        # Import komponen standar dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, log_to_ui
        
        # Setup environment dan load konfigurasi
        env, config = setup_notebook_environment("env_config")
        
        # Buat komponen UI
        ui_components = create_env_config_ui(env, config)
        
        # Setup logging dengan arahkan ke UI
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger: 
            ui_components['logger'] = logger
            logger.info("🚀 Inisialisasi environment config dimulai")
        
        # Jalankan inisialisasi konfigurasi di thread terpisah
        def init_configs_async():
            try:
                # Verifikasi konfigurasi default
                verify_default_configs(ui_components)
                
                # Setup Drive sync initializer (diutamakan)
                try:
                    from smartcash.common.drive_sync_initializer import initialize_configs
                    success, message = initialize_configs(logger)
                    if logger: logger.info(f"🔄 Sinkronisasi konfigurasi: {message}")
                except ImportError:
                    # Fallback ke versi UI jika common tidak tersedia
                    try:
                        from smartcash.ui.setup.drive_sync_initializer import initialize_configs
                        success, message = initialize_configs(logger)
                        if logger: logger.info(f"🔄 Sinkronisasi konfigurasi UI: {message}")
                    except ImportError as e:
                        if logger: logger.debug(f"ℹ️ Drive sync initializer tidak tersedia: {str(e)}")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat inisialisasi konfigurasi: {str(e)}")
        
        # Jalankan inisialisasi konfigurasi di thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            config_future = executor.submit(init_configs_async)
        
        # Setup handlers
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Integrasi handler dengan observer
        if 'observer_manager' in ui_components:
            setup_observer_integration(ui_components)
        
        # Log selesai inisialisasi
        if logger: logger.info("✅ Environment config selesai diinisialisasi")
    
    except Exception as e:
        # Fallback sederhana jika terjadi error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui, show_status
        ui_components = create_fallback_ui(ui_components, f"❌ Error saat inisialisasi environment config: {str(e)}", "error")
        show_status(f"Error: {str(e)}", "error", ui_components)
    
    return ui_components

def verify_default_configs(ui_components: Dict[str, Any]) -> None:
    """Verifikasi dan pastikan konfigurasi default tersedia dengan one-liner."""
    logger = ui_components.get('logger')
    
    try:
        import os, yaml, json, shutil
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor
        
        # Setup direktori configs
        config_dir = Path('configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # List konfigurasi default yang diperlukan
        required_configs = ["base_config.yaml", "colab_config.yaml", "dataset_config.yaml", 
                            "preprocessing_config.yaml", "training_config.yaml"]
        
        # Fungsi pemrosesan file konfigurasi
        def process_config_file(config_file):
            config_path = config_dir / config_file
            if not config_path.exists():
                # Buat template sederhana
                basic_config = {
                    '_base_': 'base_config.yaml' if config_file != "base_config.yaml" else None,
                    'created_date': 'auto-generated',
                    'config_type': config_file.replace('_config.yaml', '')
                }
                with open(config_path, 'w') as f:
                    yaml.dump(basic_config, f, default_flow_style=False)
                return config_file
            return None
        
        # Proses pembuatan file konfigurasi secara paralel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_config_file, required_configs))
        
        # Filter None dan dapatkan daftar file yang dibuat
        created_files = [f for f in results if f]
        
        if created_files and logger:
            logger.info(f"✅ Membuat {len(created_files)} file konfigurasi default")
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat verifikasi konfigurasi: {str(e)}")

def setup_observer_integration(ui_components: Dict[str, Any]) -> None:
    """Setup integrasi observer dengan handlers konfigurasi environment."""
    observer_manager = ui_components.get('observer_manager')
    logger = ui_components.get('logger')
    
    if not observer_manager:
        return
    
    try:
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Events yang relevan untuk konfigurasi environment
        events = [
            EventTopics.CONFIG_UPDATED,
            EventTopics.CONFIG_LOADED,
            EventTopics.CONFIG_RESET,
            EventTopics.CONFIG_ERROR
        ]
        
        # Handler untuk events konfigurasi
        def config_event_handler(event_type, sender, message=None, **kwargs):
            if not message:
                return
                
            # Log ke UI
            status = "error" if event_type == EventTopics.CONFIG_ERROR else \
                     "success" if event_type == EventTopics.CONFIG_UPDATED else "info"
            
            # Log ke UI dan logger
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, message, status)
            
            if logger:
                if status == "error": logger.error(message)
                elif status == "success": logger.success(message)
                else: logger.info(message)
        
        # Register observers
        for event in events:
            observer_manager.create_simple_observer(
                event_type=event,
                callback=config_event_handler,
                name=f"EnvConfig_{event}_Observer",
                group="env_config_observers"
            )
            
        if logger:
            logger.debug(f"👁️ Observer integration berhasil disetup")
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup observer integration: {str(e)}")