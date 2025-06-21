from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.setup.dependency.handlers.config_extractor import extract_dependency_config
from smartcash.ui.setup.dependency.handlers.config_updater import update_dependency_ui
from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config

class DependencyConfigHandler(ConfigHandler):
    """Fixed ConfigHandler dengan public config access dan generator cleanup"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan caching"""
        self._current_config = extract_dependency_config(ui_components)
        return self._current_config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan caching"""
        self._current_config = config.copy()
        update_dependency_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        return get_default_dependency_config()
    
    def load_config(self, config_name: str = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan caching - required by CommonInitializer"""
        try:
            config = super().load_config(config_name, use_base_config)
            self._current_config = config.copy()
            return config
        except Exception as e:
            self.logger.warning(f"⚠️ Load config error: {str(e)}")
            default_config = self.get_default_config()
            self._current_config = default_config.copy()
            return default_config
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config"""
        return self._current_config.copy()