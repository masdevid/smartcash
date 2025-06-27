
# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/config_handler.py - FIXED
# Deskripsi: Config handler konsisten dengan pattern preprocessing
# =============================================================================

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from .config_extractor import extract_dependency_config
from .config_updater import update_dependency_ui
from .defaults import get_default_dependency_config

class DependencyConfigHandler(ConfigHandler):
    """Config handler untuk dependency management konsisten dengan preprocessing pattern"""
    
    def __init__(self):
        super().__init__('dependency', 'setup')
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI - delegate ke config_extractor"""
        return extract_dependency_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config - delegate ke config_updater"""
        update_dependency_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py"""
        return get_default_dependency_config()