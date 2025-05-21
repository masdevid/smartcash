"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment
"""

from pathlib import Path
from IPython.display import display, HTML
import sys
import os
from typing import Dict, Any

from smartcash.common.config import get_config_manager
from smartcash.common.utils import is_colab
from smartcash.ui.setup.env_config.components.manager_setup import setup_managers
from smartcash.ui.utils.ui_logger import create_ui_logger
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.log_accordion import create_log_accordion

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi konfigurasi environment
    
    Returns:
        Dictionary UI components
    """
    try:
        # Buat komponen UI dasar
        ui_components = {}
        ui_components['header'] = create_header("SmartCash Environment Configuration", 
                                              "Konfigurasi lingkungan untuk SmartCash")
        
        # Gunakan komponen log_accordion standar yang sudah ada
        log_panel = create_log_accordion("Log Konfigurasi Environment")
        ui_components['log_accordion'] = log_panel['log_accordion']
        ui_components['log_output'] = log_panel['log_output']
        ui_components['ui'] = ui_components['log_accordion']
        
        # Setup logger
        logger = create_ui_logger(ui_components, "env_config")
        ui_components['logger'] = logger
        
        # Tampilkan UI
        display(ui_components['header'])
        display(ui_components['log_accordion'])
        
        # Setup config managers dan direktori
        config_manager, base_dir, config_dir = setup_managers(ui_components)
        
        # Log informasi konfigurasi
        logger.success("Environment berhasil dikonfigurasi")
        logger.info(f"📁 Base directory: {base_dir}")
        logger.info(f"📁 Config directory: {config_dir}")
        
        # Cek apakah config_dir adalah symlink
        if config_dir.is_symlink():
            target = Path(config_dir).resolve()
            logger.info(f"🔗 Config directory adalah symlink ke: {target}")
            
            # Verifikasi bahwa symlink berfungsi
            if not target.exists():
                logger.warning(f"Target symlink tidak ditemukan: {target}")
                logger.info("Mencoba memperbaiki symlink...")
                # Perbaikan akan ditangani oleh setup_managers
        
        # Tampilkan informasi environment sistem
        logger.info(f"📊 Informasi Environment:")
        logger.info(f"🐍 Python version: {sys.version.split()[0]}")
        
        # Cek apakah sedang berjalan di Colab
        colab_status = "Ya" if is_colab() else "Tidak"
        logger.info(f"💻 Running di Google Colab: {colab_status}")
        
        # Jika di Colab, tampilkan informasi konfigurasi Colab
        if is_colab():
            try:
                # Ambil konfigurasi Colab
                colab_config = config_manager.get_config('colab')
                
                if colab_config:
                    # Tampilkan informasi drive
                    if 'drive' in colab_config:
                        drive_config = colab_config['drive']
                        logger.info(f"🗄️ Pengaturan Google Drive:")
                        logger.info(f"- Sinkronisasi aktif: {drive_config.get('use_drive', False)}")
                        logger.info(f"- Strategi sinkronisasi: {drive_config.get('sync_strategy', 'none')}")
                        logger.info(f"- Gunakan symlinks: {drive_config.get('symlinks', False)}")
                        
                        # Tampilkan paths jika ada
                        if 'paths' in drive_config:
                            paths = drive_config['paths']
                            logger.info(f"- SmartCash dir: {paths.get('smartcash_dir', 'SmartCash')}")
                            logger.info(f"- Configs dir: {paths.get('configs_dir', 'configs')}")
                    
                    # Tampilkan informasi model jika menggunakan GPU/TPU
                    if 'model' in colab_config:
                        model_config = colab_config['model']
                        logger.info(f"⚡ Pengaturan Hardware:")
                        logger.info(f"- Gunakan GPU: {model_config.get('use_gpu', False)}")
                        logger.info(f"- Gunakan TPU: {model_config.get('use_tpu', False)}")
                        logger.info(f"- Precision: {model_config.get('precision', 'float32')}")
                    
                    # Informasi performa
                    if 'performance' in colab_config:
                        perf_config = colab_config['performance']
                        logger.info(f"🚀 Pengaturan Performa:")
                        logger.info(f"- Auto garbage collect: {perf_config.get('auto_garbage_collect', False)}")
                        logger.info(f"- Simpan checkpoint ke Drive: {perf_config.get('checkpoint_to_drive', False)}")
            except Exception as e:
                logger.warning(f"Gagal memuat konfigurasi Colab: {str(e)}")
        
        # Tampilkan konfigurasi file yang tersedia
        try:
            # Filter config yang tidak perlu dilaporkan jika tidak ada
            ignored_configs = ['inference', 'export', 'environment']
            
            available_configs = config_manager.get_available_configs(ignored_configs)
            
            if available_configs:
                logger.info(f"📝 File Konfigurasi Tersedia:")
                for config in available_configs:
                    logger.info(f"- {config}")
            else:
                logger.warning("Tidak ada file konfigurasi yang ditemukan.")
                logger.info("💡 Tip: Pastikan direktori konfigurasi berisi file .yaml")
        except Exception as e:
            logger.error(f"Gagal mendapatkan daftar konfigurasi: {str(e)}")
        
        # Return UI components dan config manager
        ui_components['config_manager'] = config_manager
        return ui_components
        
    except Exception as e:
        # Jika exception terjadi saat ui_components belum dibuat
        if 'ui_components' not in locals() or not ui_components:
            ui_components = {}
            ui_components['header'] = create_header("SmartCash Environment Configuration", 
                                                  "Error saat inisialisasi environment", 
                                                  is_error=True)
            
            # Gunakan komponen log_accordion standar yang sudah ada
            log_panel = create_log_accordion("Log Error")
            ui_components['log_accordion'] = log_panel['log_accordion']
            ui_components['log_output'] = log_panel['log_output']
            
            display(ui_components['header'])
            display(ui_components['log_accordion'])
            
            # Setup logger jika belum ada
            logger = create_ui_logger(ui_components, "env_config")
        
        # Log error
        logger.error(f"Error saat inisialisasi environment: {str(e)}")
        
        # Coba dapatkan config manager sebagai fallback
        try:
            config_manager = get_config_manager()
            ui_components['config_manager'] = config_manager
        except:
            pass
        
        return ui_components
