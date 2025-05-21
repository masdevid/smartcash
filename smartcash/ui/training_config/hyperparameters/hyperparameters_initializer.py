"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_initializer.py
Deskripsi: Inisialisasi komponen UI untuk konfigurasi hyperparameter
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from pathlib import Path
import os

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.utils.header_utils import create_header
from smartcash.common.logger import get_logger, LogLevel
from smartcash.common.config import get_config_manager
from smartcash.common.environment import get_environment_manager

from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import get_default_base_dir
from smartcash.ui.training_config.hyperparameters.handlers.config_writer import update_ui_from_config
from smartcash.ui.training_config.hyperparameters.handlers.info_panel_updater import update_hyperparameters_info
from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import setup_hyperparameters_button_handlers
from smartcash.ui.training_config.hyperparameters.handlers.form_handlers import setup_hyperparameters_form_handlers
from smartcash.ui.training_config.hyperparameters.components.main_components import create_hyperparameters_ui_components

# Setup logger dengan level INFO untuk mengurangi log berlebihan
logger = get_logger(__name__)
logger.set_level(LogLevel.INFO)

def initialize_hyperparameters_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi dan tampilkan UI untuk konfigurasi hyperparameter.
    
    Args:
        env: Environment manager
        config: Konfigurasi hyperparameter
        
    Returns:
        Dict berisi komponen UI
    """
    try:
        # Dapatkan base directory
        base_dir = get_default_base_dir()
        
        # Dapatkan config manager
        config_manager = get_config_manager(base_dir=base_dir)
        
        # Coba dapatkan UI components yang sudah teregistrasi
        ui_components = None
        try:
            ui_components = config_manager.get_ui_components('hyperparameters')
        except (AttributeError, Exception):
            # config_manager mungkin tidak memiliki metode get_ui_components
            logger.info("Metode get_ui_components tidak tersedia, membuat UI components baru")
            ui_components = None
        
        if ui_components:
            logger.info("Menggunakan UI components yang sudah teregistrasi")
            
            # Update UI dari konfigurasi jika ada perubahan
            try:
                current_config = config_manager.get_module_config('hyperparameters', {})
                if current_config != ui_components.get('config', {}):
                    logger.info("Memperbarui UI dari konfigurasi yang berubah")
                    update_ui_from_config(ui_components, current_config)
                    ui_components['config'] = current_config
                    
                    # Update info panel
                    if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                        ui_components['update_hyperparameters_info'](ui_components)
            except Exception as e:
                logger.warning(f"Gagal memperbarui UI dari konfigurasi: {str(e)}")
        else:
            # Jika tidak ada, inisialisasi UI baru
            logger.info("Inisialisasi UI components baru")
            
            # Dapatkan environment manager jika belum tersedia
            env = env or get_environment_manager(base_dir=base_dir)
            
            # Validasi config
            if config is None:
                logger.info("Mengambil konfigurasi dari config manager")
                config = config_manager.get_module_config('hyperparameters', {})
                
                # Jika config masih kosong, gunakan default
                if not config:
                    logger.info("Konfigurasi kosong, menggunakan default")
                    config = get_default_hyperparameters_config()
            
            # Buat komponen UI
            logger.info("Membuat komponen UI")
            ui_components = create_hyperparameters_ui_components()
            
            # Simpan referensi ke fungsi update_hyperparameters_info
            ui_components['update_hyperparameters_info'] = update_hyperparameters_info
            
            # Setup button handlers
            logger.info("Setup button handlers")
            ui_components = setup_hyperparameters_button_handlers(ui_components, env, config)
            
            # Setup form handlers
            logger.info("Setup form handlers")
            ui_components = setup_hyperparameters_form_handlers(ui_components, env, config)
            
            # Update UI dari config
            logger.info("Memperbarui UI dari konfigurasi")
            update_ui_from_config(ui_components, config)
            
            # Simpan konfigurasi saat ini ke ui_components
            ui_components['config'] = config
            
            # Simpan UI components ke config manager jika metode tersedia
            try:
                logger.info("Mendaftarkan UI components ke config manager")
                config_manager.register_ui_components('hyperparameters', ui_components)
            except (AttributeError, Exception):
                logger.info("Metode register_ui_components tidak tersedia, mengabaikan")
        
        # Tampilkan layout utama
        if 'main_layout' in ui_components:
            logger.info("Menampilkan layout utama")
            display(ui_components['main_layout'])
        elif 'main_container' in ui_components:
            logger.info("Menampilkan main container")
            display(ui_components['main_container'])
        else:
            logger.error("Layout utama tidak ditemukan")
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Layout utama tidak ditemukan",
                alert_type='error'
            ))
        
        logger.info("UI hyperparameter berhasil diinisialisasi")
        return ui_components
        
    except Exception as e:
        logger.error(f"Error saat inisialisasi UI hyperparameter: {str(e)}")
        display(create_info_alert(
            f"{ICONS.get('error', '❌')} Error saat inisialisasi UI hyperparameter: {str(e)}",
            alert_type='error'
        ))
        raise