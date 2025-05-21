"""
File: smartcash/ui/training_config/training_strategy/training_strategy_initializer.py
Deskripsi: Initializer untuk UI konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import ipywidgets as widgets
from pathlib import Path
import os

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager

from smartcash.ui.training_config.training_strategy.components import create_training_strategy_ui_components
from smartcash.ui.training_config.training_strategy.handlers import (
    update_ui_from_config, 
    update_config_from_ui, 
    get_default_config,
    update_training_strategy_info,
    get_training_strategy_config,
    setup_training_strategy_button_handlers,
    setup_training_strategy_form_handlers,
    add_status_panel,
    update_status_panel,
    get_default_base_dir
)

logger = get_logger(__name__)

def initialize_training_strategy_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi dan tampilkan komponen UI untuk konfigurasi strategi pelatihan.
    
    Args:
        env: Environment manager
        config: Konfigurasi strategi pelatihan
        
    Returns:
        Dict berisi komponen UI
    """
    try:
        logger.info(f"{ICONS.get('info', 'ℹ️')} Inisialisasi UI strategi pelatihan")
        
        # Dapatkan environment manager jika diperlukan
        if env is None:
            try:
                from smartcash.common.environment import get_environment_manager
                env = get_environment_manager(base_dir=get_default_base_dir())
            except ImportError:
                logger.warning(f"{ICONS.get('warning', '⚠️')} Environment manager tidak tersedia")
                
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Dapatkan konfigurasi jika belum tersedia
        if config is None:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Mengambil konfigurasi strategi pelatihan")
            config = get_training_strategy_config()
            if not config:
                logger.info(f"{ICONS.get('info', 'ℹ️')} Konfigurasi strategi pelatihan tidak ditemukan, menggunakan default")
                config = get_default_config()
        
        # Buat komponen UI
        logger.info(f"{ICONS.get('info', 'ℹ️')} Membuat komponen UI")
        ui_components = create_training_strategy_ui_components()
        
        # Tambahkan status panel
        logger.info(f"{ICONS.get('info', 'ℹ️')} Menambahkan status panel")
        ui_components = add_status_panel(ui_components)
        
        # Tambahkan referensi ke config
        ui_components['config'] = config
        
        # Update UI dari konfigurasi
        logger.info(f"{ICONS.get('info', 'ℹ️')} Memperbarui UI dari konfigurasi")
        update_ui_from_config(ui_components, config)
        
        # Setup handler untuk tombol
        logger.info(f"{ICONS.get('info', 'ℹ️')} Setup button handlers")
        ui_components = setup_training_strategy_button_handlers(ui_components, env, config)
        
        # Setup handler untuk form
        logger.info(f"{ICONS.get('info', 'ℹ️')} Setup form handlers")
        ui_components = setup_training_strategy_form_handlers(ui_components, env, config)
        
        # Simpan referensi ke fungsi update_training_strategy_info jika belum ada
        if 'update_training_strategy_info' not in ui_components:
            ui_components['update_training_strategy_info'] = lambda comp=ui_components: update_training_strategy_info(comp)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Mendaftarkan UI components ke config manager")
            # Register ui_components ke registry global
            if hasattr(config_manager, 'register_ui_components'):
                config_manager.register_ui_components('training_strategy', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {str(persist_error)}")
        
        # Tampilkan UI secara otomatis
        logger.info(f"{ICONS.get('info', 'ℹ️')} Menampilkan UI")
        clear_output(wait=True)
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
            
        # Update status panel dengan pesan selamat datang
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan siap digunakan", "info")
        
        # Update info panel
        if 'update_training_strategy_info' in ui_components and callable(ui_components['update_training_strategy_info']):
            ui_components['update_training_strategy_info'](ui_components)
        
        logger.info(f"{ICONS.get('success', '✅')} UI strategi pelatihan berhasil diinisialisasi")
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI strategi pelatihan: {str(e)}")
        
        # Buat UI minimal jika terjadi error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi UI strategi pelatihan</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        
        return {'main_container': error_container, 'error': str(e)}

def get_training_strategy_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mendapatkan komponen UI untuk konfigurasi strategi pelatihan.
    
    Args:
        env: Environment manager
        config: Konfigurasi strategi pelatihan
        
    Returns:
        Dict berisi komponen UI
    """
    config_manager = get_config_manager(base_dir=get_default_base_dir())
    
    # Coba dapatkan UI components yang sudah teregistrasi
    ui_components = None
    try:
        if hasattr(config_manager, 'get_ui_components'):
            ui_components = config_manager.get_ui_components('training_strategy')
    except Exception as e:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Tidak dapat mengambil UI components yang teregistrasi: {str(e)}")
    
    if ui_components and 'main_container' in ui_components:
        # Update UI dari konfigurasi jika ada perubahan
        try:
            current_config = get_training_strategy_config()
            stored_config = ui_components.get('config', {})
            
            if current_config != stored_config:
                logger.info(f"{ICONS.get('info', 'ℹ️')} Konfigurasi berubah, memperbarui UI")
                update_ui_from_config(ui_components, current_config)
                ui_components['config'] = current_config
                # Update info panel
                if 'update_training_strategy_info' in ui_components and callable(ui_components['update_training_strategy_info']):
                    ui_components['update_training_strategy_info'](ui_components)
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memperbarui UI dari konfigurasi: {str(e)}")
        
        return ui_components
    
    # Jika tidak ada, inisialisasi UI baru
    return initialize_training_strategy_ui(env, config)