"""
File: smartcash/ui/setup/dependency/handlers/config_handler.py
Deskripsi: Fixed DependencyConfigHandler dengan proper constructor dan error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.common.logger import get_logger, safe_log_to_ui

class DependencyConfigHandler(BaseConfigHandler):
    """Fixed ConfigHandler dengan default parameters dan proper error handling"""
    
    def __init__(self, module_name: str = 'dependency', parent_module: str = 'setup'):
        """Constructor dengan default parameters untuk menghindari missing args error"""
        super().__init__(module_name, parent_module)
        self._current_config = {}
        self.logger = get_logger(f"smartcash.ui.{parent_module}.{module_name}")
        self._ui_components = None
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan safe fallback"""
        try:
            from .config_extractor import extract_dependency_config
            self._current_config = extract_dependency_config(ui_components)
            return self._current_config
        except Exception as e:
            self.logger.warning(f"⚠️ Extract config error: {str(e)}")
            self._current_config = self.get_default_config()
            return self._current_config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan safe error handling"""
        try:
            from .config_updater import update_dependency_ui
            self._current_config = config.copy()
            update_dependency_ui(ui_components, config)
            self._log_to_ui("🔄 Dependency config updated", "success", ui_components)
        except Exception as e:
            self.logger.error(f"❌ Update UI error: {str(e)}")
            self._log_to_ui(f"❌ Update error: {str(e)}", "error", ui_components)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback yang aman"""
        try:
            from .defaults import get_default_dependency_config
            return get_default_dependency_config()
        except ImportError:
            self.logger.info("📋 Using fallback default config")
            return {
                'module_name': 'dependency',
                'dependencies': {
                    'torch': {'version': 'latest', 'required': True},
                    'torchvision': {'version': 'latest', 'required': True},
                    'ultralytics': {'version': 'latest', 'required': True}
                },
                'install_options': {
                    'force_reinstall': False,
                    'upgrade': True,
                    'quiet': False
                }
            }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config"""
        return self._current_config.copy()
    
    def _log_to_ui(self, message: str, level: str, ui_components: Optional[Dict[str, Any]] = None):
        """Safe logging ke UI components"""
        target_ui = ui_components or self._ui_components
        if target_ui:
            safe_log_to_ui(target_ui, message, level)
        
        # Log ke standard logger juga
        if level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components reference untuk logging"""
        self._ui_components = ui_components