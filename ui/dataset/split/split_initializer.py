"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Initializer untuk UI konfigurasi split dataset dengan arsitektur logger dan config yang diperbaharui
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.ui_logger import create_ui_logger


def initialize_split_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi split dataset dengan arsitektur yang diperbaharui.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi untuk dataset (optional)
        
    Returns:
        Dict berisi komponen UI
    """
    # Setup komponen dasar
    ui_components = {
        'module_name': 'dataset_split',
        'logger_namespace': 'smartcash.ui.dataset.split'
    }
    
    # Setup UI logger dengan namespace
    output_widget = widgets.Output()
    ui_components['log_output'] = output_widget
    logger = create_ui_logger(ui_components, 'split_config', redirect_stdout=False)
    ui_components['logger'] = logger
    
    try:
        # Setup environment dan config manager
        env = env or get_environment_manager()
        config_manager = get_config_manager(base_dir=str(env.base_dir))
        ui_components['config_manager'] = config_manager
        ui_components['env'] = env
        
        # Load atau buat konfigurasi default
        config = config or _load_or_create_config(config_manager, logger)
        
        # Buat komponen UI
        ui_components.update(_create_split_components(config, logger))
        
        # Setup handlers
        _setup_handlers(ui_components, logger)
        
        # Tampilkan UI
        display(ui_components['main_container'])
        
        return ui_components
        
    except Exception as e:
        logger.error(f"💥 Error inisialisasi split UI: {str(e)}")
        _display_error_fallback(str(e))
        return ui_components


def _load_or_create_config(config_manager, logger) -> Dict[str, Any]:
    """Load atau buat konfigurasi default untuk split dataset."""
    try:
        config = config_manager.get_config('dataset_config')
        if not config or 'data' not in config:
            from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
            config = get_default_split_config()
            config_manager.save_config(config, 'dataset_config')
            logger.info("📋 Menggunakan konfigurasi split default")
        return config
    except Exception as e:
        logger.warning(f"⚠️ Error loading config, gunakan default: {str(e)}")
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        return get_default_split_config()


def _create_split_components(config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Buat komponen UI untuk split dataset."""
    try:
        from smartcash.ui.dataset.split.components.split_form import create_split_form
        from smartcash.ui.dataset.split.components.split_layout import create_split_layout
        
        # Buat form komponen
        form_components = create_split_form(config)
        
        # Buat layout utama
        layout_components = create_split_layout(form_components)
        
        return {**form_components, **layout_components}
        
    except Exception as e:
        logger.error(f"💥 Error membuat komponen UI: {str(e)}")
        return {'main_container': widgets.HTML(f"<div style='color:red'>Error: {str(e)}</div>")}


def _setup_handlers(ui_components: Dict[str, Any], logger) -> None:
    """Setup event handlers untuk UI components."""
    try:
        from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
        from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
        
        # Setup button handlers
        setup_button_handlers(ui_components)
        
        # Setup slider handlers untuk auto-adjustment
        setup_slider_handlers(ui_components)
        
        logger.debug("🔗 Event handlers berhasil dipasang")
        
    except Exception as e:
        logger.error(f"💥 Error setup handlers: {str(e)}")


def _display_error_fallback(error_message: str) -> None:
    """Tampilkan error fallback jika inisialisasi gagal."""
    error_widget = widgets.HTML(
        value=f"""
        <div style='color: #dc3545; padding: 15px; border: 1px solid #dc3545; 
                   border-radius: 5px; background-color: #f8d7da; margin: 10px 0;'>
            <h4>❌ Error Inisialisasi Split Dataset UI</h4>
            <p><strong>Detail:</strong> {error_message}</p>
            <p><em>Silakan restart cell atau periksa konfigurasi sistem.</em></p>
        </div>
        """
    )
    display(error_widget)


def create_split_config_cell():
    """Fungsi helper untuk membuat cell konfigurasi split dataset."""
    return initialize_split_ui()