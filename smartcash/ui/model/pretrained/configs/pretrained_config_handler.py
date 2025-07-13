"""
File: smartcash/ui/model/pretrained/configs/pretrained_config_handler.py
Description: Configuration handler for pretrained module following UIModule pattern
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.logger import get_module_logger
from .pretrained_defaults import get_default_pretrained_config


class PretrainedConfigHandler(ConfigHandler):
    """
    Configuration handler for pretrained module.
    
    Features:
    - 📋 Configuration validation and merging
    - 🔄 UI-to-config and config-to-UI synchronization
    - ✅ Model configuration validation
    - 💾 Configuration persistence and loading
    - 🛡️ Error handling and validation rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pretrained configuration handler.
        
        Args:
            config: Optional initial configuration
        """
        super().__init__(
            module_name='pretrained',
            default_config=get_default_pretrained_config()
        )
        
        self.logger = get_module_logger("smartcash.ui.model.pretrained.config")
        
        # Load initial configuration
        if config:
            self.update_config(config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate pretrained configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required sections
            if 'pretrained' not in config:
                self.logger.error("Missing required section: 'pretrained'")
                return False
            
            pretrained_config = config['pretrained']
            
            # Check required fields
            required_fields = ['models_dir', 'model_urls']
            for field in required_fields:
                if field not in pretrained_config:
                    self.logger.error(f"Missing required field: pretrained.{field}")
                    return False
            
            # Validate models_dir
            models_dir = pretrained_config['models_dir']
            if not isinstance(models_dir, str) or not models_dir.strip():
                self.logger.error("models_dir must be a non-empty string")
                return False
            
            # Validate model_urls
            model_urls = pretrained_config['model_urls']
            if not isinstance(model_urls, dict):
                self.logger.error("model_urls must be a dictionary")
                return False
            
            # Validate timeout if present
            if 'download_timeout' in pretrained_config:
                timeout = pretrained_config['download_timeout']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    self.logger.error("download_timeout must be a positive number")
                    return False
            
            # Validate chunk_size if present
            if 'chunk_size' in pretrained_config:
                chunk_size = pretrained_config['chunk_size']
                if not isinstance(chunk_size, int) or chunk_size <= 0:
                    self.logger.error("chunk_size must be a positive integer")
                    return False
            
            self.logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            form_components = ui_components.get('form_components', {})
            
            # Extract configuration from form components
            config = {
                'pretrained': {
                    'models_dir': self._get_widget_value(form_components, 'models_dir', '/data/pretrained'),
                    'model_urls': {
                        'yolov5s': self._get_widget_value(form_components, 'yolov5s_url', ''),
                        'efficientnet_b4': self._get_widget_value(form_components, 'efficientnet_b4_url', '')
                    },
                    'auto_download': self._get_widget_value(form_components, 'auto_download', False),
                    'validate_downloads': self._get_widget_value(form_components, 'validate_downloads', True),
                    'cleanup_failed': self._get_widget_value(form_components, 'cleanup_failed', True),
                    'download_timeout': self._get_widget_value(form_components, 'download_timeout', 300),
                    'chunk_size': self._get_widget_value(form_components, 'chunk_size', 8192),
                    'progress_update_interval': self._get_widget_value(form_components, 'progress_update_interval', 1024 * 1024)
                },
                'models': {
                    'yolov5s': {
                        'enabled': self._get_widget_value(form_components, 'yolov5s_enabled', True),
                        'priority': self._get_widget_value(form_components, 'yolov5s_priority', 1),
                        'expected_size': self._get_widget_value(form_components, 'yolov5s_expected_size', 14_400_000),
                        'validation': self._get_widget_value(form_components, 'yolov5s_validation', True)
                    },
                    'efficientnet_b4': {
                        'enabled': self._get_widget_value(form_components, 'efficientnet_b4_enabled', True),
                        'priority': self._get_widget_value(form_components, 'efficientnet_b4_priority', 2),
                        'expected_size': self._get_widget_value(form_components, 'efficientnet_b4_expected_size', 75_000_000),
                        'validation': self._get_widget_value(form_components, 'efficientnet_b4_validation', True)
                    }
                },
                'operations': {
                    'download': {
                        'enabled': self._get_widget_value(form_components, 'download_enabled', True),
                        'concurrent': self._get_widget_value(form_components, 'download_concurrent', False),
                        'retry_count': self._get_widget_value(form_components, 'download_retry_count', 3),
                        'verify_integrity': self._get_widget_value(form_components, 'download_verify_integrity', True)
                    },
                    'validate': {
                        'enabled': self._get_widget_value(form_components, 'validate_enabled', True),
                        'check_size': self._get_widget_value(form_components, 'validate_check_size', True),
                        'check_format': self._get_widget_value(form_components, 'validate_check_format', True)
                    },
                    'cleanup': {
                        'enabled': self._get_widget_value(form_components, 'cleanup_enabled', True),
                        'remove_corrupted': self._get_widget_value(form_components, 'cleanup_remove_corrupted', True),
                        'backup_before_delete': self._get_widget_value(form_components, 'cleanup_backup_before_delete', False)
                    }
                },
                'ui': {
                    'show_progress': self._get_widget_value(form_components, 'show_progress', True),
                    'auto_refresh': self._get_widget_value(form_components, 'auto_refresh', True),
                    'confirm_cleanup': self._get_widget_value(form_components, 'confirm_cleanup', True)
                }
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error extracting config from UI: {e}")
            return self.get_config()
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components to update
            config: Configuration to apply (uses current config if None)
        """
        try:
            config = config or self.get_config()
            form_components = ui_components.get('form_components', {})
            
            # Update pretrained configuration
            pretrained_config = config.get('pretrained', {})
            self._set_widget_value(form_components, 'models_dir', pretrained_config.get('models_dir'))
            self._set_widget_value(form_components, 'auto_download', pretrained_config.get('auto_download'))
            self._set_widget_value(form_components, 'validate_downloads', pretrained_config.get('validate_downloads'))
            self._set_widget_value(form_components, 'cleanup_failed', pretrained_config.get('cleanup_failed'))
            self._set_widget_value(form_components, 'download_timeout', pretrained_config.get('download_timeout'))
            self._set_widget_value(form_components, 'chunk_size', pretrained_config.get('chunk_size'))
            self._set_widget_value(form_components, 'progress_update_interval', pretrained_config.get('progress_update_interval'))
            
            # Update model URLs
            model_urls = pretrained_config.get('model_urls', {})
            self._set_widget_value(form_components, 'yolov5s_url', model_urls.get('yolov5s'))
            self._set_widget_value(form_components, 'efficientnet_b4_url', model_urls.get('efficientnet_b4'))
            
            # Update models configuration
            models_config = config.get('models', {})
            yolov5s_config = models_config.get('yolov5s', {})
            self._set_widget_value(form_components, 'yolov5s_enabled', yolov5s_config.get('enabled'))
            self._set_widget_value(form_components, 'yolov5s_priority', yolov5s_config.get('priority'))
            self._set_widget_value(form_components, 'yolov5s_expected_size', yolov5s_config.get('expected_size'))
            self._set_widget_value(form_components, 'yolov5s_validation', yolov5s_config.get('validation'))
            
            efficientnet_config = models_config.get('efficientnet_b4', {})
            self._set_widget_value(form_components, 'efficientnet_b4_enabled', efficientnet_config.get('enabled'))
            self._set_widget_value(form_components, 'efficientnet_b4_priority', efficientnet_config.get('priority'))
            self._set_widget_value(form_components, 'efficientnet_b4_expected_size', efficientnet_config.get('expected_size'))
            self._set_widget_value(form_components, 'efficientnet_b4_validation', efficientnet_config.get('validation'))
            
            # Update operations configuration
            operations_config = config.get('operations', {})
            download_config = operations_config.get('download', {})
            self._set_widget_value(form_components, 'download_enabled', download_config.get('enabled'))
            self._set_widget_value(form_components, 'download_concurrent', download_config.get('concurrent'))
            self._set_widget_value(form_components, 'download_retry_count', download_config.get('retry_count'))
            self._set_widget_value(form_components, 'download_verify_integrity', download_config.get('verify_integrity'))
            
            validate_config = operations_config.get('validate', {})
            self._set_widget_value(form_components, 'validate_enabled', validate_config.get('enabled'))
            self._set_widget_value(form_components, 'validate_check_size', validate_config.get('check_size'))
            self._set_widget_value(form_components, 'validate_check_format', validate_config.get('check_format'))
            
            cleanup_config = operations_config.get('cleanup', {})
            self._set_widget_value(form_components, 'cleanup_enabled', cleanup_config.get('enabled'))
            self._set_widget_value(form_components, 'cleanup_remove_corrupted', cleanup_config.get('remove_corrupted'))
            self._set_widget_value(form_components, 'cleanup_backup_before_delete', cleanup_config.get('backup_before_delete'))
            
            # Update UI configuration
            ui_config = config.get('ui', {})
            self._set_widget_value(form_components, 'show_progress', ui_config.get('show_progress'))
            self._set_widget_value(form_components, 'auto_refresh', ui_config.get('auto_refresh'))
            self._set_widget_value(form_components, 'confirm_cleanup', ui_config.get('confirm_cleanup'))
            
            self.logger.debug("UI updated from configuration successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating UI from config: {e}")
    
    def _get_widget_value(self, form_components: Dict[str, Any], widget_name: str, default_value: Any) -> Any:
        """Get value from widget with fallback to default."""
        try:
            widget = form_components.get(widget_name)
            if widget and hasattr(widget, 'value'):
                return widget.value
            return default_value
        except Exception:
            return default_value
    
    def _set_widget_value(self, form_components: Dict[str, Any], widget_name: str, value: Any) -> None:
        """Set widget value safely."""
        try:
            widget = form_components.get(widget_name)
            if widget and hasattr(widget, 'value') and value is not None:
                widget.value = value
        except Exception as e:
            self.logger.debug(f"Could not set {widget_name} value: {e}")