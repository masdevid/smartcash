# File: smartcash/ui/pretrained/handlers/config_handler.py
"""
File: smartcash/ui/pretrained/handlers/config_handler.py
Deskripsi: Config handler untuk pretrained models dengan konsistensi pattern dan DRY utilities
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.pretrained.handlers.config_extractor import extract_pretrained_config
from smartcash.ui.pretrained.handlers.config_updater import update_pretrained_ui
from smartcash.common.config.manager import get_config_manager

if TYPE_CHECKING:
    from smartcash.common.logger import LoggerBridge
else:
    LoggerBridge = Any  # For runtime type hints

class PretrainedConfigHandler(ConfigHandler):
    """Pretrained config handler dengan consistent pattern dan error handling"""
    
    def __init__(self, module_name: str = 'pretrained', parent_module: str = 'ui'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'pretrained_config.yaml'
        self._ui_components: Optional[Dict[str, Any]] = None
        self._logger_bridge: Optional[LoggerBridge] = None
    
    @property
    def logger_bridge(self) -> Optional[LoggerBridge]:
        """Get logger bridge dari UI components atau fallback ke parent logger"""
        if self._logger_bridge:
            return self._logger_bridge
        if self._ui_components and 'logger_bridge' in self._ui_components:
            self._logger_bridge = self._ui_components['logger_bridge']
            return self._logger_bridge
        return None
        
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components and update logger bridge reference"""
        self._ui_components = ui_components
        if 'logger_bridge' in ui_components:
            self._logger_bridge = ui_components['logger_bridge']
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'debug'):
            self.logger_bridge.debug(message, **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'info'):
            self.logger_bridge.info(message, **kwargs)
            
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'warning'):
            self.logger_bridge.warning(message, **kwargs)
            
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'error'):
            self.logger_bridge.error(message, exc_info=exc_info, **kwargs)
            
    def _log_operation_success(self, message: str) -> None:
        """Log successful operation with consistent formatting"""
        self._log_info(message)
        
    def _log_operation_error(self, message: str) -> None:
        """Log operation error with consistent formatting"""
        self._log_error(message, exc_info=True)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI dengan enhanced error handling"""
        try:
            config = extract_pretrained_config(ui_components)
            self._log_operation_success("🔧 Config extracted successfully")
            return config
        except Exception as e:
            error_msg = f"❌ Error extracting config: {str(e)}"
            self._log_operation_error(error_msg)
            raise  # Re-raise to parent CommonInitializer error handler
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan config dan error handling"""
        try:
            update_pretrained_ui(ui_components, config)
            self._log_operation_success("✅ UI updated with config")
        except Exception as e:
            error_msg = f"❌ Error updating UI: {str(e)}"
            self._log_operation_error(error_msg)
            raise
    
    def _log_operation_success(self, message: str) -> None:
        """Log successful operation dengan proper logger"""
        self.logger_bridge.info(message) if hasattr(self.logger_bridge, 'info') else self.logger.info(message)
    
    def _log_operation_error(self, message: str) -> None:
        """Log error operation dengan proper logger"""
        self.logger_bridge.error(message) if hasattr(self.logger_bridge, 'error') else self.logger.error(message)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk pretrained models"""
        try:
            from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
            return get_default_pretrained_config()
        except ImportError as e:
            logger.error(f"❌ Defaults module not found: {str(e)}")
            raise  # Re-raise to parent CommonInitializer error handler
    
# Remove fallback config - errors should propagate to CommonInitializer