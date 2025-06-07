
from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui

class PreprocessingConfigHandler(ConfigHandler):
    """Config handler untuk preprocessing dengan fixed implementation"""
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'  # Explicitly use dataset_config.yaml

    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari preprocessing UI components"""
        extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_preprocessing_ui(ui_components, config)
