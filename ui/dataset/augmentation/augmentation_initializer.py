"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Inisialisasi antarmuka augmentasi dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger

def initialize_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Inisialisasi antarmuka augmentasi dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI
    """
    try:
        logger = get_logger('augmentation')
        logger.info(f"🚀 Memulai inisialisasi UI augmentasi dataset")
        
        # Import komponen UI
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        # Buat komponen UI
        ui_components = create_augmentation_ui(env, config)
        
        # Tambahkan logger
        ui_components['logger'] = logger
        
        # Setup handler
        logger.info(f"🔄 Setup handlers untuk UI augmentasi")
        setup_handlers(ui_components, env, config)
        
        # Inisialisasi UI dari konfigurasi
        if 'update_ui_from_config' in ui_components and callable(ui_components['update_ui_from_config']):
            logger.info(f"⚙️ Memperbarui UI dari konfigurasi")
            try:
                ui_components['update_ui_from_config'](ui_components, config)
            except Exception as e:
                logger.warning(f"⚠️ Error saat memperbarui UI dari konfigurasi: {str(e)}")
                import traceback
                logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Pastikan UI persisten
        try:
            from smartcash.ui.dataset.augmentation.handlers.persistence_handler import ensure_ui_persistence
            logger.info(f"💾 Memastikan persistensi UI")
            ensure_ui_persistence(ui_components)
        except Exception as e:
            logger.warning(f"⚠️ Error saat memastikan persistensi UI: {str(e)}")
        
        # Update informasi augmentasi
        try:
            from smartcash.ui.dataset.augmentation.handlers.status_handler import update_augmentation_info
            logger.info(f"ℹ️ Memperbarui informasi augmentasi")
            update_augmentation_info(ui_components)
        except Exception as e:
            logger.warning(f"⚠️ Error saat memperbarui informasi augmentasi: {str(e)}")
        
        logger.info(f"✅ Inisialisasi UI augmentasi selesai")
        return ui_components
    except Exception as e:
        logger = get_logger('augmentation')
        logger.error(f"❌ Error fatal saat inisialisasi UI augmentasi: {str(e)}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Buat UI minimal sebagai fallback
        import ipywidgets as widgets
        from IPython.display import display
        
        error_ui = widgets.VBox([
            widgets.HTML(f"<h2 style='color: red;'>⚠️ Error saat memuat UI Augmentasi Dataset</h2>"),
            widgets.HTML(f"<p>Detail error: {str(e)}</p>"),
            widgets.Button(description="Coba Lagi", button_style="danger")
        ])
        
        # Tambahkan handler untuk tombol coba lagi
        error_ui.children[-1].on_click(lambda b: create_and_display_augmentation_ui(env, config))
        
        return {'ui': error_ui, 'logger': logger}

def setup_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk antarmuka augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    # Import handler
    from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
    from smartcash.ui.dataset.augmentation.handlers.config_handler import update_config_from_ui, update_ui_from_config
    from smartcash.ui.dataset.augmentation.handlers.initialization_handler import register_progress_callback, reset_progress_bar
    
    # Setup handler
    ui_components = setup_button_handlers(ui_components, env, config)
    
    # Tambahkan referensi ke handler
    ui_components['update_config_from_ui'] = update_config_from_ui
    ui_components['update_ui_from_config'] = update_ui_from_config
    ui_components['register_progress_callback'] = register_progress_callback
    ui_components['reset_progress_bar'] = reset_progress_bar
    
    return ui_components

def display_augmentation_ui(ui_components: Dict[str, Any]) -> None:
    """
    Tampilkan antarmuka augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Pastikan ui_components memiliki kunci 'ui'
        if 'ui' not in ui_components:
            logger = ui_components.get('logger', get_logger('augmentation'))
            logger.error("❌ Komponen UI tidak ditemukan dalam ui_components")
            raise KeyError("Komponen 'ui' tidak ditemukan dalam ui_components")
        
        # Tampilkan UI
        display(ui_components['ui'])
        
        # Log sukses
        if 'logger' in ui_components:
            ui_components['logger'].info("✅ UI augmentasi berhasil ditampilkan")
    except Exception as e:
        logger = ui_components.get('logger', get_logger('augmentation'))
        logger.error(f"❌ Error saat menampilkan UI augmentasi: {str(e)}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")

def create_and_display_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat dan tampilkan antarmuka augmentasi dataset.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary komponen UI
    """
    try:
        # Inisialisasi UI
        ui_components = initialize_augmentation_ui(env, config)
        
        # Tampilkan UI
        display_augmentation_ui(ui_components)
        
        return ui_components
    except Exception as e:
        logger = get_logger('augmentation')
        logger.error(f"❌ Error fatal saat membuat dan menampilkan UI augmentasi: {str(e)}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        # Buat UI minimal sebagai fallback
        import ipywidgets as widgets
        from IPython.display import display
        
        error_ui = widgets.VBox([
            widgets.HTML(f"<h2 style='color: red;'>⚠️ Error saat memuat UI Augmentasi Dataset</h2>"),
            widgets.HTML(f"<p>Detail error: {str(e)}</p>"),
            widgets.Button(description="Coba Lagi", button_style="danger")
        ])
        
        # Tambahkan handler untuk tombol coba lagi
        def retry_handler(b):
            try:
                # Hapus UI error
                display(widgets.HTML("<p>Mencoba memuat ulang UI...</p>"))
                # Coba lagi
                create_and_display_augmentation_ui(env, config)
            except Exception as retry_error:
                logger.error(f"❌ Error saat mencoba ulang: {str(retry_error)}")
                display(widgets.HTML(f"<p style='color: red;'>Gagal memuat ulang: {str(retry_error)}</p>"))
        
        error_ui.children[-1].on_click(retry_handler)
        
        # Tampilkan UI error
        display(error_ui)
        
        return {'ui': error_ui, 'logger': logger}
