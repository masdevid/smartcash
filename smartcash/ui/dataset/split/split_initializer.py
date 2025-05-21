"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Initializer untuk UI konfigurasi split dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display

def initialize_split_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi split dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk dataset
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': 'dataset_split'}
    
    # Setup UI logger
    from smartcash.ui.utils.ui_logger import create_ui_logger
    output_widget = widgets.Output()
    ui_components['output_log'] = output_widget
    logger = create_ui_logger(ui_components, 'split_config')
    ui_components['logger'] = logger
    
    try:
        # Import dependency
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config.manager import get_config_manager
        from smartcash.ui.dataset.split.handlers.sync_logger import add_sync_status_panel
        
        # Dapatkan environment jika belum tersedia
        env = env or get_environment_manager()
        
        # Ensure base_dir is set
        if not getattr(env, 'base_dir', None):
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, "base_dir tidak ditemukan. Pastikan direktori base valid untuk konfigurasi.")
            raise ValueError("base_dir must not be None. Please provide a valid base directory for configuration.")
        
        # Dapatkan config manager
        config_manager = get_config_manager(base_dir=env.base_dir, config_file='dataset_config.yaml')
        
        # Load konfigurasi dari config manager
        if config is None:
            # Dapatkan konfigurasi dari config manager atau file
            try:
                from smartcash.ui.dataset.split.handlers.config_handlers import load_config
                config = config_manager.get_module_config('dataset') if config_manager else load_config()
            except Exception as e:
                from smartcash.ui.dataset.split.handlers.config_handlers import get_default_split_config
                config = get_default_split_config()
        
        # Buat komponen UI
        try:
            from smartcash.ui.dataset.split.components.split_components import create_split_ui
            ui_components.update(create_split_ui(config))
        except Exception as ui_error:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, f"Error saat membuat komponen UI: {str(ui_error)}")
            raise
        
        # Tambahkan panel status sinkronisasi
        try:
            ui_components = add_sync_status_panel(ui_components)
        except Exception as panel_error:
            logger.warning(f"⚠️ Error saat menambahkan panel status: {str(panel_error)}")
        
        # Setup button handlers
        try:
            from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
            ui_components = setup_button_handlers(ui_components, config, env)
        except Exception as handler_error:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, f"Error saat setup button handlers: {str(handler_error)}")
            raise
        
        # Deteksi apakah berjalan di Colab untuk keperluan debugging
        try:
            from smartcash.ui.dataset.split.handlers.config_handlers import is_colab_environment
            is_colab = is_colab_environment()
            if is_colab:
                logger.info("🔍 Terdeteksi berjalan di lingkungan Google Colab")
        except Exception as colab_error:
            logger.warning(f"⚠️ Error saat mendeteksi lingkungan Colab: {str(colab_error)}")
        
        # Tampilkan UI
        display(ui_components['ui'])
        
        return ui_components
    
    except Exception as e:
        # Import log_sync_error jika belum di-import
        try:
            from smartcash.ui.dataset.split.handlers.sync_logger import log_sync_error
            log_sync_error(ui_components, f"Error saat inisialisasi UI: {str(e)}")
        except:
            pass
            
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        # Tampilkan pesan error
        error_widget = widgets.HTML(
            value=f"<div style='color: red; padding: 10px; border: 1px solid red;'><b>Error:</b> {str(e)}</div>"
        )
        display(error_widget)
        
        # Kembalikan komponen minimal
        return ui_components

def create_split_config_cell():
    """
    Buat cell untuk konfigurasi split dataset.
    
    Fungsi ini digunakan untuk membuat cell yang dapat dijalankan di notebook.
    """
    # Inisialisasi UI
    initialize_split_ui()
