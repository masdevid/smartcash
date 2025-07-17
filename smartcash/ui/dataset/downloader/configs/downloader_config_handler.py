"""
File: smartcash/ui/dataset/downloader/configs/downloader_config_handler.py
Description: Config handler for dataset downloader module using core mixins.
"""

from typing import Dict, Any, Optional, Type, TypeVar

from smartcash.ui.core.mixins.configuration_mixin import ConfigurationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.logger import get_module_logger
from .downloader_defaults import get_default_downloader_config

T = TypeVar('T', bound='DownloaderConfigHandler')

class DownloaderConfigHandler(LoggingMixin, ConfigurationMixin):
    """
    Config handler for dataset downloader module.
    
    Uses LoggingMixin for logging and ConfigurationMixin for configuration management.
    Follows the mixin pattern used throughout the UI module system.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize downloader config handler.
        
        Args:
            default_config: Optional default configuration dictionary
        """
        # Initialize mixins
        super().__init__()
        
        # Module identification for logging
        self.module_name = 'downloader'
        self.parent_module = 'dataset'
        
        # Store default config
        self._default_config = default_config or get_default_downloader_config()
        self._api_key_initialized = False
        
        # Initialize internal config directly instead of calling _initialize_config_handler
        # to avoid infinite recursion
        self._merged_config = self._default_config.copy()
        
        self.log("DownloaderConfigHandler initialized", 'debug')
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return self._default_config.copy()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'DownloaderConfigHandler':
        """Create a new instance of this config handler."""
        # Return self to avoid infinite recursion
        return self
    
    # ==================== CONFIGURATION METHODS ====================
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_filename: Optional config filename (default: 'dataset_config.yaml')
            
        Returns:
            Result dictionary with success status and config
        """
        try:
            # Load default config and initialize API key if needed
            config = self.get_default_config()
            
            if not config.get('data', {}).get('roboflow', {}).get('api_key'):
                self._initialize_api_key()
                config = self._merged_config.copy()
                
            return {'success': True, 'config': config}
        except Exception as e:
            self.log(f"Error loading config: {e}", 'error')
            return {'success': False, 'config': self.get_default_config(), 'error': str(e)}
    
    def save_config(self, ui_components: Dict[str, Any] = None, config_filename: str = None) -> Dict[str, Any]:
        """
        Save configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            config_filename: Optional config filename
            
        Returns:
            Result dictionary with success status
        """
        try:
            if ui_components:
                extracted_config = self.extract_config(ui_components)
                if extracted_config:
                    self._merged_config.update(extracted_config)
                    
            return {'success': True, 'message': 'Configuration saved successfully'}
        except Exception as e:
            self.log(f"Error saving config: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            # Delegate to centralized extractor to avoid duplication
            from .downloader_extractor import extract_downloader_config
            return extract_downloader_config(ui_components)
        except Exception as e:
            self.log(f"Error extracting config: {e}", 'error')
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with status and messages
        """
        try:
            # Delegate to centralized validator to avoid duplication
            from .downloader_updater import validate_ui_inputs
            # Convert config to ui_components format for validation
            ui_components = self._config_to_ui_components(config)
            return validate_ui_inputs(ui_components)
        except Exception as e:
            self.log(f"Error validating config: {e}", 'error')
            return {'valid': False, 'errors': [str(e)]}
    
    def _initialize_api_key(self) -> None:
        """
        Initialize API key from secrets manager.
        
        This is called lazily when the API key is first needed.
        """
        if self._api_key_initialized:
            return
            
        try:
            from smartcash.ui.core.mixins.colab_secrets_mixin import ColabSecretsMixin
            
            secrets_mixin = ColabSecretsMixin()
            api_key = secrets_mixin.get_api_key()
            
            if api_key:
                # Update merged config directly
                self._merged_config.setdefault('data', {}).setdefault('roboflow', {})['api_key'] = api_key
                
            self._api_key_initialized = True
            self.log("API key initialized", 'debug')
            
        except Exception as e:
            self.log(f"Failed to load API key: {e}", 'warning')
            self._api_key_initialized = True

    def _config_to_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config dictionary to UI components format for validation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            UI components dictionary
        """
        roboflow = config.get('data', {}).get('roboflow', {})
        download = config.get('download', {})
        
        # Mock UI components for validation
        class MockWidget:
            def __init__(self, value):
                self.value = value
        
        return {
            'workspace_input': MockWidget(roboflow.get('workspace', '')),
            'project_input': MockWidget(roboflow.get('project', '')),
            'version_input': MockWidget(roboflow.get('version', '')),
            'api_key_input': MockWidget(roboflow.get('api_key', '')),
            'validate_checkbox': MockWidget(download.get('validate_download', True)),
            'backup_checkbox': MockWidget(download.get('backup_existing', False))
        }

def get_downloader_config_handler() -> DownloaderConfigHandler:
    """
    Get or create a downloader config handler instance.
    
    Returns:
        DownloaderConfigHandler instance
    """
    return DownloaderConfigHandler()