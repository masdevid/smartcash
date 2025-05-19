"""
File: smartcash/ui/setup/env_config/handlers/sync_handler.py
Deskripsi: Handler untuk sinkronisasi konfigurasi
"""

from typing import Dict, Any
import os
from pathlib import Path

from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.ui_logger import log_to_ui

def sync_configs(ui_components: Dict[str, Any], env_manager: Any, logger: Any) -> None:
    """
    Sinkronisasi konfigurasi dengan Google Drive
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
        logger: Logger
    """
    # Periksa environment tanpa log berlebihan
    env_info = env_manager.get_system_info()
    
    # Pastikan direktori configs ada
    configs_dir = Path(env_manager.base_dir) / 'configs'
    os.makedirs(configs_dir, exist_ok=True)
    
    # Sinkronisasi konfigurasi dengan Google Drive jika terhubung
    if hasattr(env_manager, 'sync_config'):
        sync_success, sync_message = env_manager.sync_config()
        if not sync_success:
            logger.warning(f"⚠️ {sync_message}")
    
    # Simpan konfigurasi environment
    if hasattr(env_manager, 'save_environment_config'):
        save_success, save_message = env_manager.save_environment_config()
        if not save_success:
            logger.warning(f"⚠️ {save_message}")
    
    # Sinkronisasi semua file konfigurasi di direktori configs
    try:
        # Tampilkan status di log output
        ui_components['status_panel'].value = create_info_box(
            "Sinkronisasi Konfigurasi", 
            "Sedang menyinkronkan semua file konfigurasi...",
            style="info"
        ).value
        
        # Gunakan force_sync_all_configs untuk memastikan semua file konfigurasi berhasil disinkronkan
        try:
            # Import fungsi force_sync_all_configs
            from smartcash.common.config.force_sync import force_sync_all_configs
            
            # Jalankan force_sync_all_configs
            sync_results = force_sync_all_configs(logger)
            
            # Log hasil sinkronisasi
            success_count = len(sync_results.get('success', []))
            skipped_count = len(sync_results.get('skipped', []))
            failure_count = len(sync_results.get('failure', []))
            
            logger.info(f"🔄 Sinkronisasi konfigurasi: {success_count} berhasil, {skipped_count} dilewati, {failure_count} gagal")
            
            # Jika ada kegagalan, coba gunakan sync_all_configs sebagai fallback
            if failure_count > 0:
                logger.warning(f"⚠️ Beberapa file gagal disinkronkan dengan force_sync_all_configs, mencoba dengan sync_all_configs...")
                
                # Import fungsi sync_all_configs
                from smartcash.common.config.sync import sync_all_configs
                
                # Jalankan sync_all_configs dengan strategi drive_priority
                fallback_results = sync_all_configs(sync_strategy='drive_priority', logger=logger)
                
                # Log hasil sinkronisasi fallback
                fallback_success_count = len(fallback_results.get('success', []))
                fallback_skipped_count = len(fallback_results.get('skipped', []))
                fallback_failure_count = len(fallback_results.get('failure', []))
                
                logger.info(f"🔄 Sinkronisasi fallback: {fallback_success_count} berhasil, {fallback_skipped_count} dilewati, {fallback_failure_count} gagal")
        except ImportError:
            # Jika force_sync_all_configs tidak tersedia, gunakan sync_all_configs
            logger.warning("⚠️ force_sync_all_configs tidak tersedia, menggunakan sync_all_configs...")
            
            # Import fungsi sync_all_configs
            from smartcash.common.config.sync import sync_all_configs
            
            # Jalankan sync_all_configs dengan strategi drive_priority
            sync_results = sync_all_configs(sync_strategy='drive_priority', logger=logger)
            
            # Log hasil sinkronisasi
            success_count = len(sync_results.get('success', []))
            skipped_count = len(sync_results.get('skipped', []))
            failure_count = len(sync_results.get('failure', []))
            
            logger.info(f"🔄 Sinkronisasi konfigurasi: {success_count} berhasil, {skipped_count} dilewati, {failure_count} gagal")
        
        # Update status panel dengan hasil sinkronisasi
        ui_components['status_panel'].value = create_info_box(
            "Sinkronisasi Konfigurasi", 
            f"Sinkronisasi konfigurasi selesai: {success_count} berhasil, {skipped_count} dilewati, {failure_count} gagal",
            style="success" if failure_count == 0 else "warning"
        ).value
    except Exception as e:
        # Log error
        logger.error(f"❌ Error saat sinkronisasi konfigurasi: {str(e)}")
        raise

def run_sync_tests(ui_components: Dict[str, Any], logger: Any) -> None:
    """
    Jalankan test sinkronisasi konfigurasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger: Logger
    """
    try:
        from smartcash.common.config.tests.test_config_sync import TestConfigSync
        
        # Update status panel
        ui_components['status_panel'].value = create_info_box(
            "Test Sinkronisasi Konfigurasi", 
            "Sedang menjalankan test sinkronisasi konfigurasi...",
            style="info"
        ).value
        
        # Tangkap output test
        import io
        import sys
        from contextlib import redirect_stdout
        
        test_output = io.StringIO()
        with redirect_stdout(test_output):
            test_result = TestConfigSync.test_config_sync()
        
        # Tampilkan hasil test di log
        test_output_str = test_output.getvalue()
        for line in test_output_str.split('\n'):
            if line.strip():
                if '⚠️' in line:
                    logger.warning(line)
                else:
                    logger.info(line)
        
        # Update status panel dengan hasil test
        ui_components['status_panel'].value = create_info_box(
            "Test Sinkronisasi Konfigurasi", 
            "Test sinkronisasi konfigurasi selesai" + (" dengan sukses" if test_result else " dengan masalah"),
            style="success" if test_result else "warning"
        ).value
    except Exception as e:
        logger.warning(f"⚠️ Error saat menjalankan test sinkronisasi konfigurasi: {str(e)}")
        # Tidak perlu update status panel karena error hanya pada test
