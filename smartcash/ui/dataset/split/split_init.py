"""
File: smartcash/ui/dataset/split/split_init.py
Deskripsi: Config cell untuk split dataset dengan arsitektur yang diperbaharui menggunakan ConfigCellInitializer
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer, create_config_cell
from smartcash.ui.dataset.split.components.split_form import create_split_form
from smartcash.ui.dataset.split.components.split_layout import create_split_layout



class SplitConfigInitializer(ConfigCellInitializer):
    """Config cell initializer untuk split dataset configuration"""
    
    def __init__(self, module_name='split_dataset', config_filename='dataset_config'):
        super().__init__(module_name, config_filename)
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk split config"""
        from smartcash.ui.dataset.split.components.split_form import create_split_form
        from smartcash.ui.dataset.split.components.split_layout import create_split_layout
        from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
        
        form_components = create_split_form(config)
        layout_components = create_split_layout(form_components)
        ui_components = {**form_components, **layout_components}
        
        # Setup custom slider handlers
        setup_slider_handlers(ui_components)
        
        return ui_components
    
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from smartcash.ui.dataset.split.handlers.ui_extractor import extract_split_config
        return extract_split_config(ui_components)
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        from smartcash.ui.dataset.split.handlers.ui_updater import update_split_ui
        update_split_ui(ui_components, config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default split config"""
        from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config
        return get_default_split_config()


def create_split_config_cell(env=None, config=None, **kwargs):
    """Factory function untuk create split config cell"""
    ui = create_config_cell(SplitConfigInitializer, env, config, **kwargs)
    return ui


# Backward compatibility aliases
initialize_split_ui = create_split_config_cell
create_split_init = create_split_config_cell