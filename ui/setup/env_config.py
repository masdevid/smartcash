"""
File: smartcash/ui/setup/env_config.py
Deskripsi: Koordinator utama untuk konfigurasi environment SmartCash dengan fallback terpusat
"""

from typing import Dict, Any
from IPython.display import display

def setup_environment_config():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi fallback_utils"""
    # Inisialisasi ui_components dengan nilai default
    ui_components = {'status': None, 'module_name': 'env_config'}
    
    try:
        # Import modul dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, log_to_ui
        from smartcash.ui.utils.fallback_utils import try_operation, create_fallback_ui

        # Setup environment dan komponen UI
        env, config = setup_notebook_environment("env_config")
        ui_components = create_env_config_ui(env, config)
        
        # Buat fallback UI jika diperlukan
        if 'ui' not in ui_components: ui_components = create_fallback_ui(ui_components, "Membuat fallback UI", "info")
        
        # Setup logging dan log inisialisasi
        if 'status' in ui_components: log_to_ui(ui_components, "🚀 Inisialisasi environment config dimulai", "info")
        logger = setup_ipython_logging(ui_components, "env_config")
        if logger: ui_components['logger'] = logger; logger.info("✅ Logger environment config berhasil diinisialisasi")
        
        # Jalankan operasi konfigurasi dengan error handling yang ditingkatkan
        try_operation(lambda: ensure_default_configs(logger), logger, "verifikasi konfigurasi default", ui_components)
        try_operation(lambda: sync_configs_with_drive(logger), logger, "sinkronisasi Drive", ui_components)
        
        # Setup handlers dan cleanup
        ui_components = setup_env_config_handlers(ui_components, env, config)
        register_cleanup(ui_components, logger)
        
    except Exception as e:
        # Gunakan create_fallback_ui dari fallback_utils
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, f"❌ Error saat inisialisasi environment config: {str(e)}", "error")
    
    return ui_components

def ensure_default_configs(logger):
    """Pastikan konfigurasi default tersedia"""
    try:
        from smartcash.common.default_config import ensure_all_configs_exist
        return ensure_all_configs_exist()
    except ImportError:
        if logger: logger.warning("⚠️ Module default_config tidak tersedia")
        return None

def sync_configs_with_drive(logger):
    """Sinkronisasi konfigurasi dengan Google Drive"""
    try:
        from smartcash.common.config_sync import sync_all_configs
        results = sync_all_configs(sync_strategy='drive_priority', create_backup=True)
        
        if logger:
            success_count = len(results.get('success', []))
            skipped_count = len(results.get('skipped', []))
            failure_count = len(results.get('failure', []))
            logger.info(f"🔄 Sinkronisasi selesai: {success_count} sukses, {skipped_count} dilewati, {failure_count} gagal")
        
        return results
    except (ImportError, TypeError):
        if logger: logger.warning("⚠️ Module config_sync tidak tersedia atau error")
        return None

def register_cleanup(ui_components, logger):
    """Daftarkan fungsi cleanup untuk resources"""
    # Definisikan fungsi cleanup
    def cleanup_resources():
        if 'observer_manager' in ui_components and 'observer_group' in ui_components:
            try: ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            except Exception as e: 
                if logger: logger.debug(f"⚠️ Error saat unregister observer: {str(e)}")
        
        try:
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
        except: pass
        
        if logger: logger.debug("🧹 Resources dibersihkan")
    
    # Daftarkan ke ui_components dan IPython
    ui_components['cleanup'] = cleanup_resources
    try:
        from IPython import get_ipython
        if get_ipython(): get_ipython().events.register('pre_run_cell', cleanup_resources)
    except Exception: pass