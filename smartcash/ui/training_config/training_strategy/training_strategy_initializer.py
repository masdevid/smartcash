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
from smartcash.common.environment import get_environment_manager

from smartcash.ui.training_config.training_strategy.components import create_training_strategy_ui_components
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import (
    update_ui_from_config, 
    update_config_from_ui, 
    get_default_config,
    update_training_strategy_info
)
from smartcash.ui.training_config.training_strategy.handlers.button_handlers import setup_training_strategy_button_handlers
from smartcash.ui.training_config.training_strategy.handlers.form_handlers import setup_training_strategy_form_handlers
from smartcash.ui.training_config.training_strategy.handlers.status_handlers import add_status_panel, update_status_panel

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

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
        logger.info("Inisialisasi UI strategi pelatihan")
        
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager(base_dir=get_default_base_dir())
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        
        # Dapatkan konfigurasi jika belum tersedia
        if config is None:
            logger.info("Mengambil konfigurasi strategi pelatihan")
            config = config_manager.get_module_config('training_strategy', {})
            if not config:
                logger.info("Konfigurasi strategi pelatihan tidak ditemukan, menggunakan default")
                config = get_default_config()
        
        # Buat komponen UI
        logger.info("Membuat komponen UI")
        ui_components = create_training_strategy_ui_components()
        
        # Tambahkan status panel
        logger.info("Menambahkan status panel")
        ui_components = add_status_panel(ui_components)
        
        # Tambahkan referensi ke config
        ui_components['config'] = config
        
        # Setup handler untuk tombol
        logger.info("Setup button handlers")
        ui_components = setup_training_strategy_button_handlers(ui_components, env, config)
        
        # Setup handler untuk form
        logger.info("Setup form handlers")
        ui_components = setup_training_strategy_form_handlers(ui_components, env, config)
        
        # Update UI dari konfigurasi
        logger.info("Memperbarui UI dari konfigurasi")
        update_ui_from_config(ui_components, config)
        
        # Simpan referensi ke fungsi update_training_strategy_info
        ui_components['update_training_strategy_info'] = update_training_strategy_info
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            logger.info("Mendaftarkan UI components ke config manager")
            config_manager.register_ui_components('training_strategy', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan UI secara otomatis
        logger.info("Menampilkan UI")
        clear_output(wait=True)
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
            
        # Update status panel dengan pesan selamat datang
        update_status_panel(ui_components, "Konfigurasi strategi pelatihan siap digunakan", "info")
        
        logger.info("UI strategi pelatihan berhasil diinisialisasi")
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
    ui_components = config_manager.get_ui_components('training_strategy')
    
    if ui_components:
        # Update UI dari konfigurasi jika ada perubahan
        current_config = config_manager.get_module_config('training_strategy', {})
        if current_config != ui_components.get('config', {}):
            update_ui_from_config(ui_components, current_config)
            ui_components['config'] = current_config
        
        return ui_components
    
    # Jika tidak ada, inisialisasi UI baru
    return initialize_training_strategy_ui(env, config)
