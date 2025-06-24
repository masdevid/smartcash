"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: Config cell untuk split dataset dengan arsitektur yang diperbaharui menggunakan ConfigCellInitializer
"""

import sys
import traceback
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.handlers.config_handlers import ConfigHandler

class SplitConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk split dataset configuration"""
    
    def __init__(self, module_name='split_dataset', config_filename='dataset_config', config_handler_class=None, 
                 parent_module: Optional[str] = None):
        super().__init__(module_name, config_filename, config_handler_class, parent_module)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk split config"""
        try:
            from smartcash.ui.dataset.split.components.ui_form import create_split_form
            from smartcash.ui.dataset.split.components.ui_layout import create_split_layout
            from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
            
            form_components = create_split_form(config)
            layout_components = create_split_layout(form_components)
            ui_components = {**form_components, **layout_components}
            
            # Setup custom slider handlers
            setup_slider_handlers(ui_components)
            
            return ui_components
            
        except Exception as e:
            return self.handle_ui_exception(e, context="UI split dataset configuration")


class SplitConfigHandler(ConfigHandler):
    """Config handler untuk split dataset configuration"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from smartcash.ui.dataset.split.handlers.config_extractor import extract_split_config
        return extract_split_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari config"""
        from smartcash.ui.dataset.split.handlers.config_updater import update_split_ui
        update_split_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default split configuration"""
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        return get_default_split_config()


def create_split_config_cell(env=None, config=None, parent_module=None, parent_callbacks=None, **kwargs):
    """
    Factory function untuk create split config cell
    
    Args:
        env: Environment manager instance
        config: Override config values
        parent_module: Parent module name (e.g., 'dataset', 'training')
        parent_callbacks: Callbacks for parent modules
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    return create_config_cell(
        SplitConfigInitializer, 
        'split_dataset', 
        'dataset_config', 
        env=env, 
        config=config, 
        config_handler_class=SplitConfigHandler,
        parent_module=parent_module,
        parent_callbacks=parent_callbacks,
        **kwargs
    )


# Backward compatibility aliases
initialize_split_ui = create_split_config_cell
create_split_init = create_split_config_cell