"""
File: smartcash/ui/training_config/training_strategy/training_strategy_initializer.py
Deskripsi: Initializer untuk UI konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
import ipywidgets as widgets

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.common.environment import get_environment_manager

from smartcash.ui.training_config.training_strategy.components import create_training_strategy_ui_components
from smartcash.ui.training_config.training_strategy.handlers.config_handlers import update_ui_from_config, update_config_from_ui, get_default_config
from smartcash.ui.training_config.training_strategy.handlers.button_handlers import setup_training_strategy_button_handlers
from smartcash.ui.training_config.training_strategy.handlers.form_handlers import setup_training_strategy_form_handlers

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
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi strategi pelatihan jika belum tersedia
        if config is None:
            config = config_manager.get_module_config('training_strategy', {})
            # Jika masih tidak ada, gunakan default
            if not config:
                config = get_default_config()
        
        # Buat komponen UI
        ui_components = create_training_strategy_ui_components()
        
        # Tambahkan referensi ke config
        ui_components['config'] = config
        
        # Setup handler untuk tombol
        ui_components = setup_training_strategy_button_handlers(ui_components, env, config)
        
        # Setup handler untuk form
        ui_components = setup_training_strategy_form_handlers(ui_components, env, config)
        
        # Update UI dari konfigurasi
        update_ui_from_config(ui_components, config)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            config_manager.register_ui_components('training_strategy', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan UI secara otomatis
        clear_output(wait=True)
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI strategi pelatihan: {str(e)}")
        
        # Buat UI minimal jika terjadi error
        error_output = widgets.Output()
        with error_output:
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat inisialisasi UI strategi pelatihan: {str(e)}",
                alert_type='error'
            ))
        
        return {'main_container': error_output, 'error': str(e)}

def get_training_strategy_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mendapatkan komponen UI untuk konfigurasi strategi pelatihan.
    
    Args:
        env: Environment manager
        config: Konfigurasi strategi pelatihan
        
    Returns:
        Dict berisi komponen UI
    """
    # Dapatkan config manager
    config_manager = get_config_manager()
    
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
