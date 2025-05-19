"""
File: smartcash/ui/training/training_initializer.py
Deskripsi: Inisialisasi UI untuk proses training model SmartCash
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

def initialize_training_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk proses training model.
    
    Args:
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI untuk training
    """
    try:
        # Dapatkan environment manager jika belum tersedia
        env = env or get_environment_manager()
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Dapatkan logger
        logger = get_logger("training_ui")
        
        # Import komponen UI baru
        from smartcash.ui.training.components.training_component import create_training_ui
        
        # Cek apakah UI components sudah terdaftar
        ui_components = config_manager.get_ui_components('training')
        
        if ui_components:
            logger.info(f"{ICONS.get('info', 'ℹ️')} Menggunakan komponen UI training yang sudah ada")
        else:
            # Buat komponen UI baru
            logger.info(f"{ICONS.get('info', 'ℹ️')} Membuat komponen UI training baru")
            ui_components = create_training_ui(env, config)
            
            # Tambahkan logger ke komponen
            ui_components['logger'] = logger
        
        # Setup handler untuk tombol
        from smartcash.ui.training.handlers.setup_handler import setup_training_handlers
        ui_components = setup_training_handlers(ui_components, env, config)
        
        # Pastikan UI components teregistrasi untuk persistensi
        try:
            config_manager.register_ui_components('training', ui_components)
        except Exception as persist_error:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat memastikan persistensi UI: {persist_error}")
        
        # Tampilkan UI secara otomatis
        clear_output(wait=True)
        if 'main_container' in ui_components:
            display(ui_components['main_container'])
        
        return ui_components
        
    except Exception as e:
        # Fallback jika terjadi error
        logger = get_logger("training_ui")
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI training: {str(e)}")
        
        # Buat UI minimal jika terjadi error
        error_output = widgets.Output()
        with error_output:
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat inisialisasi UI training: {str(e)}",
                alert_type='error'
            ))
        
        return {'main_container': error_output, 'error': str(e)}
