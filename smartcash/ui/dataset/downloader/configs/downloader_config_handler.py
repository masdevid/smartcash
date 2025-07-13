"""
Downloader Configuration Handler - Simplified New Core Pattern
Clean implementation without backward compatibility.
"""
from typing import Dict, Any, Optional
import logging

from smartcash.ui.core.handlers import ConfigHandler
from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config

class DownloaderConfigHandler(ConfigHandler):
    """Simplified config handler for downloader module."""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset', **kwargs):
        """Initialize config handler."""
        super().__init__(module_name=module_name, parent_module=parent_module)
        self.logger = logging.getLogger(f"smartcash.ui.{parent_module}.{module_name}")
        self._config = get_default_downloader_config()
        self._api_key_initialized = False
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load configuration."""
        try:
            config = self._config.copy()
            
            # Initialize API key lazily if needed
            if not config.get('data', {}).get('roboflow', {}).get('api_key'):
                self._initialize_api_key()
                
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return get_default_downloader_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save configuration from UI components."""
        try:
            # Extract config from UI
            extracted_config = self.extract_config(ui_components)
            if extracted_config:
                self._config.update(extracted_config)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            return False
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components."""
        try:
            form_widgets = ui_components.get('form_widgets', {})
            
            config = {
                'data': {
                    'roboflow': {
                        'workspace': getattr(form_widgets.get('workspace_input'), 'value', ''),
                        'project': getattr(form_widgets.get('project_input'), 'value', ''),
                        'version': getattr(form_widgets.get('version_input'), 'value', ''),
                        'api_key': getattr(form_widgets.get('api_key_input'), 'value', ''),
                    }
                },
                'download': {
                    'validate_download': getattr(form_widgets.get('validate_checkbox'), 'value', True),
                    'backup_existing': getattr(form_widgets.get('backup_checkbox'), 'value', False),
                }
            }
            
            return config
        except Exception as e:
            self.logger.error(f"Error extracting config: {e}")
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration."""
        try:
            if not isinstance(config, dict):
                raise ValueError('Configuration must be a dictionary')
            return config
        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    def _initialize_api_key(self) -> None:
        """Initialize API key from secrets."""
        if self._api_key_initialized:
            return
            
        try:
            from smartcash.ui.dataset.downloader.services.utils.secret_manager import SecretManagerService
            secret_manager = SecretManagerService()
            api_key = secret_manager.get_api_key()
            
            if api_key:
                if 'data' not in self._config:
                    self._config['data'] = {}
                if 'roboflow' not in self._config['data']:
                    self._config['data']['roboflow'] = {}
                    
                self._config['data']['roboflow']['api_key'] = api_key
                
            self._api_key_initialized = True
        except Exception as e:
            self.logger.warning(f"Failed to load API key: {e}")
            self._api_key_initialized = True

def get_downloader_config_handler() -> DownloaderConfigHandler:
    """Get downloader config handler instance."""
    return DownloaderConfigHandler()