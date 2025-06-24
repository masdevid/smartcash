"""
File: smartcash/ui/strategy/handlers/config_handler.py
Deskripsi: Config handler untuk strategy module
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class StrategyConfigHandler(ConfigHandler):
    """Handler untuk strategy configuration"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from .config_extractor import extract_strategy_config
        return extract_strategy_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        from .config_updater import update_strategy_ui
        update_strategy_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        from .defaults import get_default_strategy_config
        return get_default_strategy_config()