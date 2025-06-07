
from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.evaluation.handlers.config_extractor import extract_config    
from smartcash.ui.evaluation.handlers.config_updater import update_ui   

class EvaluationConfigHandler(ConfigHandler):
    """Config handler untuk evaluation dengan fixed implementation"""
    
    def __init__(self, module_name: str = 'evaluation'):
        super().__init__(module_name)
        self.config_manager = get_config_manager()
        self.config_filename = 'evaluation_config.yaml'

    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari evaluation UI components"""
        extract_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_ui(ui_components, config)
   