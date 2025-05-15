"""
File: smartcash/ui/training_config/hyperparameters/hyperparameters_initializer.py
Deskripsi: Inisialisasi komponen UI untuk konfigurasi hyperparameter
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.ui.utils.header_utils import create_header
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

from smartcash.ui.training_config.hyperparameters.components.hyperparameters_components import create_hyperparameters_ui_components
from smartcash.ui.training_config.hyperparameters.handlers.config_handlers import update_ui_from_config, update_config_from_ui
from smartcash.ui.training_config.hyperparameters.handlers.button_handlers import setup_hyperparameters_button_handlers
from smartcash.ui.training_config.hyperparameters.handlers.form_handlers import setup_hyperparameters_form_handlers

logger = get_logger(__name__)

def initialize_hyperparameters_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi dan tampilkan komponen UI untuk konfigurasi hyperparameter.
    
    Args:
        env: Environment manager
        config: Konfigurasi hyperparameter
        
    Returns:
        Dict berisi komponen UI
    """
    try:
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi hyperparameter jika belum tersedia
        if config is None:
            config = config_manager.get_module_config('hyperparameters', {})
        
        # Buat komponen UI
        ui_components = create_hyperparameters_ui_components()
        
        # Tambahkan referensi ke config
        ui_components['config'] = config
        
        # Setup handler untuk tombol
        ui_components = setup_hyperparameters_button_handlers(ui_components, env, config)
        
        # Setup handler untuk form
        ui_components = setup_hyperparameters_form_handlers(ui_components, env, config)
        
        # Update UI dari konfigurasi
        update_ui_from_config(ui_components, config)
        
        # Update info panel
        if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
            ui_components['update_hyperparameters_info'](ui_components)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            config_manager.register_ui_components('hyperparameters', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan UI secara otomatis
        clear_output(wait=True)
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
        else:
            display_hyperparameters_ui(ui_components)
        
        return ui_components
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI hyperparameter: {str(e)}")
        
        # Buat UI minimal jika terjadi error
        error_output = widgets.Output()
        with error_output:
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat inisialisasi UI hyperparameter: {str(e)}",
                alert_type='error'
            ))
        
        return {'main_layout': error_output, 'error': str(e)}

def display_hyperparameters_ui(ui_components: Dict[str, Any]) -> None:
    """
    Menampilkan UI hyperparameter.
    
    Args:
        ui_components: Komponen UI
    """
    try:
        # Tampilkan header
        display(create_header(
            title="Konfigurasi Hyperparameter",
            description="Pengaturan parameter untuk proses training model",
            icon=ICONS.get('settings', '⚙️')
        ))
        
        # Tampilkan layout utama
        if 'main_layout' in ui_components:
            display(ui_components['main_layout'])
        else:
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Layout utama tidak ditemukan",
                alert_type='error'
            ))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan UI hyperparameter: {str(e)}")
        display(create_info_alert(
            f"{ICONS.get('error', '❌')} Error saat menampilkan UI hyperparameter: {str(e)}",
            alert_type='error'
        ))

def get_hyperparameters_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mendapatkan komponen UI untuk konfigurasi hyperparameter.
    
    Args:
        env: Environment manager
        config: Konfigurasi hyperparameter
        
    Returns:
        Dict berisi komponen UI
    """
    # Dapatkan config manager
    config_manager = get_config_manager()
    
    # Coba dapatkan UI components yang sudah teregistrasi
    ui_components = config_manager.get_ui_components('hyperparameters')
    
    if ui_components:
        # Update UI dari konfigurasi jika ada perubahan
        current_config = config_manager.get_module_config('hyperparameters', {})
        if current_config != ui_components.get('config', {}):
            update_ui_from_config(ui_components, current_config)
            ui_components['config'] = current_config
            
            # Update info panel
            if 'update_hyperparameters_info' in ui_components and callable(ui_components['update_hyperparameters_info']):
                ui_components['update_hyperparameters_info'](ui_components)
        
        return ui_components
    
    # Jika tidak ada, inisialisasi UI baru
    return initialize_hyperparameters_ui(env, config)
