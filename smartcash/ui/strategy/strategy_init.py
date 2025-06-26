"""
File: smartcash/ui/strategy/strategy_init.py
Deskripsi: Updated strategy initializer sesuai dengan handlers dan UI structure
"""

from typing import Dict, Any, Optional, Type, Union
from IPython.display import display
from smartcash.ui.initializers.config_cell_initializer import ConfigCellHandler, create_config_cell
from smartcash.common.logger import get_logger

logger = get_logger(__name__)
MODULE_NAME = 'strategy'
MODULE_CONFIG = f"{MODULE_NAME}_config"


class StrategyInitializer(ConfigCellInitializer):
    """Config cell initializer untuk strategy configuration"""
    
    def __init__(self, module_name: str = MODULE_NAME, config_filename: str = MODULE_CONFIG,
                 config_handler_class: Optional[Type] = None, parent_module: Optional[str] = None):
        if config_handler_class is None:
            from .handlers.config_handler import StrategyConfigHandler
            config_handler_class = StrategyConfigHandler
            
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Optional[Dict[str, Any]] = None, env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk strategy configuration
        
        Args:
            config: Konfigurasi yang akan digunakan. Jika None, akan menggunakan default config
            env: Environment untuk UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dict[str, Any]: Komponen UI yang sudah dibuat
        """
        try:
            logger.info("🎯 Memulai pembuatan strategy UI...")
            
            # Inisialisasi config jika None atau bukan dictionary
            if not isinstance(config, dict):
                logger.warning("⚠️ Config tidak valid, menggunakan default config")
                config = self.config_handler.get_default_config()
                # Asumsikan get_default_config() selalu mengembalikan dict yang valid
                # Jika tidak, biarkan exception terjadi secara alami
            
            # Create form components
            from .components.ui_form import create_strategy_form
            form_components = create_strategy_form(config)
            
            # Create layout with existing components
            from .components.ui_layout import create_strategy_layout
            layout_components = create_strategy_layout(form_components)
            
            # Merge components
            ui_components = {**form_components, **layout_components}
            
            # Setup event handlers
            from .handlers.strategy_handlers import setup_strategy_event_handlers, setup_dynamic_summary_updates
            setup_strategy_event_handlers(ui_components)
            setup_dynamic_summary_updates(ui_components)
            
            logger.info("✅ Strategy UI berhasil dibuat")
            return ui_components
            
        except Exception as e:
            logger.error(f"❌ Error creating strategy UI: {str(e)}")
            return self.handle_ui_exception(e, "creating strategy UI")


def create_strategy_config_cell(env=None, config=None, parent_module=None, **kwargs):
    """Factory function untuk strategy config cell
    
    Returns:
        Dict[str, Any]: Dictionary dengan komponen UI yang berisi minimal 'main_layout'
    """
    logger.info("🏭 Membuat strategy config cell...")
    
    try:
        # Buat initializer
        initializer = StrategyInitializer(MODULE_NAME, MODULE_CONFIG, None, parent_module)
        
        # Dapatkan komponen UI
        ui_components = initializer.initialize(env, config, **kwargs)
        
        # Pastikan return value adalah dictionary
        if not isinstance(ui_components, dict):
            logger.warning(f"⚠️ Tipe return tidak valid: {type(ui_components)}, mengkonversi ke dict")
            ui_components = {'main_layout': ui_components}
        
        # Pastikan ada main_layout
        if 'main_layout' not in ui_components:
            if 'ui_components' in ui_components:
                # Gunakan ui_components sebagai main_layout jika tersedia
                from ipywidgets import VBox
                ui_components['main_layout'] = VBox(ui_components['ui_components']) \
                    if isinstance(ui_components['ui_components'], (list, tuple)) \
                    else ui_components['ui_components']
            else:
                # Coba buat layout default dari semua widget yang ada
                from ipywidgets import VBox
                widgets = [
                    comp for comp in ui_components.values() 
                    if hasattr(comp, 'layout') or hasattr(comp, 'value')
                ]
                if widgets:
                    ui_components['main_layout'] = VBox(widgets)
                else:
                    # Jika tidak ada widget yang valid, buat pesan error
                    error_msg = "Tidak ada widget UI yang valid ditemukan"
                    logger.error(f"❌ {error_msg}")
                    ui_components['main_layout'] = create_fallback_ui(
                        error_message=error_msg,
                        module_name=MODULE_NAME,
                        traceback=error_msg
                    )
        
        # Pastikan main_layout ada dan valid
        if not ui_components.get('main_layout'):
            error_msg = "Gagal membuat layout utama untuk UI"
            logger.error(f"❌ {error_msg}")
            ui_components['main_layout'] = create_fallback_ui(
                error_message=error_msg,
                module_name=MODULE_NAME,
                traceback=error_msg
            )
        
        logger.info("✅ Strategy config cell berhasil dibuat")
        return ui_components
            
    except Exception as e:
        error_msg = f"Gagal membuat strategy config cell: {str(e)}"
        logger.error(f"❌ {error_msg}")
        fallback_ui = create_fallback_ui(
            error_message=error_msg,
            module_name=MODULE_NAME,
            traceback=str(e)
        )
        return {'main_layout': fallback_ui}