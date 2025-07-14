"""
File: smartcash/ui/dataset/split/configs/split_config_handler.py
Description: Configuration handler for split module following UIModule pattern
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.config_handler import ConfigurableHandler
from smartcash.ui.logger import get_module_logger
from .split_defaults import get_default_split_config, VALIDATION_RULES


class SplitConfigHandler(ConfigurableHandler):
    """
    Configuration handler for split module.
    
    Features:
    - 📋 Configuration validation and merging
    - 🔄 UI-to-config and config-to-UI synchronization
    - ✅ Split ratio validation with constraints
    - 💾 Configuration persistence and loading
    - 🛡️ Error handling and validation rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize split configuration handler.
        
        Args:
            config: Optional initial configuration
        """
        super().__init__(
            module_name='split',
            default_config=get_default_split_config()
        )
        
        self.logger = get_module_logger("smartcash.ui.dataset.split.config")
        self._validation_rules = VALIDATION_RULES
        self._ui_components = None  # Will be set by the module
        
        # Load initial configuration
        if config:
            self.update_config(config)
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components for config extraction and updates."""
        self._ui_components = ui_components
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate split configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If configuration violates constraints
        """
        try:
            # Check required sections
            if 'split' not in config:
                self.logger.error("Missing required section: 'split'")
                return False
            
            split_config = config['split']
            
            # Check required fields
            required_fields = ['ratios', 'seed', 'input_dir', 'output_dir']
            for field in required_fields:
                if field not in split_config:
                    self.logger.error(f"Missing required field: split.{field}")
                    return False
            
            # Validate ratios
            ratios = split_config['ratios']
            if not isinstance(ratios, dict):
                self.logger.error("Split ratios must be a dictionary")
                return False
            
            required_ratios = ['train', 'val', 'test']
            for ratio_name in required_ratios:
                if ratio_name not in ratios:
                    self.logger.error(f"Missing ratio: {ratio_name}")
                    return False
                
                ratio_value = ratios[ratio_name]
                if not isinstance(ratio_value, (int, float)):
                    self.logger.error(f"Ratio {ratio_name} must be numeric")
                    return False
                
                if not (0.0 <= ratio_value <= 1.0):
                    self.logger.error(f"Ratio {ratio_name} must be between 0.0 and 1.0")
                    return False
            
            # Check ratios sum
            ratios_sum = sum(ratios.values())
            if not (0.999 <= ratios_sum <= 1.001):
                error_msg = f"Split ratios must sum to 1.0, got {ratios_sum:.3f}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate seed
            if not isinstance(split_config['seed'], int):
                self.logger.error("Seed must be an integer")
                return False
            
            # Validate directories
            for dir_field in ['input_dir', 'output_dir']:
                if not isinstance(split_config[dir_field], str):
                    self.logger.error(f"{dir_field} must be a string")
                    return False
            
            self.logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            if not self._ui_components:
                self.logger.warning("No UI components available for config extraction")
                return self.config
                
            form_components = self._ui_components.get('form_components', {})
            
            # Extract configuration from form components
            config = {
                'split': {
                    'input_dir': self._get_widget_value(form_components, 'input_dir', 'data/raw'),
                    'output_dir': self._get_widget_value(form_components, 'output_dir', 'data/split'),
                    'ratios': {
                        'train': self._get_widget_value(form_components, 'train_ratio', 0.7),
                        'val': self._get_widget_value(form_components, 'val_ratio', 0.15),
                        'test': self._get_widget_value(form_components, 'test_ratio', 0.15)
                    },
                    'method': self._get_widget_value(form_components, 'split_method', 'random'),
                    'seed': self._get_widget_value(form_components, 'seed', 42),
                    'shuffle': self._get_widget_value(form_components, 'shuffle', True),
                    'preserve_structure': self._get_widget_value(form_components, 'preserve_structure', True),
                    'copy_files': self._get_widget_value(form_components, 'copy_files', True),
                    'create_dirs': self._get_widget_value(form_components, 'create_dirs', True)
                },
                'data': {
                    'file_extensions': self._get_widget_value(form_components, 'file_extensions', ['.jpg', '.jpeg', '.png', '.bmp']),
                    'min_files_per_split': self._get_widget_value(form_components, 'min_files_per_split', 1),
                    'validate_images': self._get_widget_value(form_components, 'validate_images', True),
                    'skip_corrupted': self._get_widget_value(form_components, 'skip_corrupted', True)
                },
                'output': {
                    'train_dir': self._get_widget_value(form_components, 'train_dir', 'train'),
                    'val_dir': self._get_widget_value(form_components, 'val_dir', 'val'),
                    'test_dir': self._get_widget_value(form_components, 'test_dir', 'test'),
                    'overwrite': self._get_widget_value(form_components, 'overwrite', False),
                    'backup': self._get_widget_value(form_components, 'backup', True),
                    'backup_dir': self._get_widget_value(form_components, 'backup_dir', 'backup')
                },
                'advanced': {
                    'use_relative_paths': self._get_widget_value(form_components, 'use_relative_paths', True),
                    'preserve_dir_structure': self._get_widget_value(form_components, 'preserve_dir_structure', True),
                    'create_symlinks': self._get_widget_value(form_components, 'create_symlinks', False),
                    'parallel_processing': self._get_widget_value(form_components, 'parallel_processing', True),
                    'batch_size': self._get_widget_value(form_components, 'batch_size', 1000),
                    'progress_interval': self._get_widget_value(form_components, 'progress_interval', 0.1)
                },
                'ui': {
                    'show_advanced': self._get_widget_value(form_components, 'show_advanced', False),
                    'auto_refresh': self._get_widget_value(form_components, 'auto_refresh', True),
                    'preview_enabled': self._get_widget_value(form_components, 'preview_enabled', True)
                }
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error extracting config from UI: {e}")
            return self.config
    
    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Update UI components from configuration.
        
        Args:
            config: Configuration to apply (uses current config if None)
        """
        try:
            if not self._ui_components:
                self.logger.warning("No UI components available for UI update")
                return
                
            config = config or self.config
            form_components = self._ui_components.get('form_components', {})
            
            # Update split configuration
            split_config = config.get('split', {})
            self._set_widget_value(form_components, 'input_dir', split_config.get('input_dir'))
            self._set_widget_value(form_components, 'output_dir', split_config.get('output_dir'))
            self._set_widget_value(form_components, 'split_method', split_config.get('method'))
            self._set_widget_value(form_components, 'seed', split_config.get('seed'))
            self._set_widget_value(form_components, 'shuffle', split_config.get('shuffle'))
            self._set_widget_value(form_components, 'preserve_structure', split_config.get('preserve_structure'))
            self._set_widget_value(form_components, 'copy_files', split_config.get('copy_files'))
            self._set_widget_value(form_components, 'create_dirs', split_config.get('create_dirs'))
            
            # Update ratios
            ratios = split_config.get('ratios', {})
            self._set_widget_value(form_components, 'train_ratio', ratios.get('train'))
            self._set_widget_value(form_components, 'val_ratio', ratios.get('val'))
            self._set_widget_value(form_components, 'test_ratio', ratios.get('test'))
            
            # Update data configuration
            data_config = config.get('data', {})
            self._set_widget_value(form_components, 'file_extensions', data_config.get('file_extensions'))
            self._set_widget_value(form_components, 'min_files_per_split', data_config.get('min_files_per_split'))
            self._set_widget_value(form_components, 'validate_images', data_config.get('validate_images'))
            self._set_widget_value(form_components, 'skip_corrupted', data_config.get('skip_corrupted'))
            
            # Update output configuration
            output_config = config.get('output', {})
            self._set_widget_value(form_components, 'train_dir', output_config.get('train_dir'))
            self._set_widget_value(form_components, 'val_dir', output_config.get('val_dir'))
            self._set_widget_value(form_components, 'test_dir', output_config.get('test_dir'))
            self._set_widget_value(form_components, 'overwrite', output_config.get('overwrite'))
            self._set_widget_value(form_components, 'backup', output_config.get('backup'))
            self._set_widget_value(form_components, 'backup_dir', output_config.get('backup_dir'))
            
            # Update advanced configuration
            advanced_config = config.get('advanced', {})
            self._set_widget_value(form_components, 'use_relative_paths', advanced_config.get('use_relative_paths'))
            self._set_widget_value(form_components, 'preserve_dir_structure', advanced_config.get('preserve_dir_structure'))
            self._set_widget_value(form_components, 'create_symlinks', advanced_config.get('create_symlinks'))
            self._set_widget_value(form_components, 'parallel_processing', advanced_config.get('parallel_processing'))
            self._set_widget_value(form_components, 'batch_size', advanced_config.get('batch_size'))
            self._set_widget_value(form_components, 'progress_interval', advanced_config.get('progress_interval'))
            
            # Update UI configuration
            ui_config = config.get('ui', {})
            self._set_widget_value(form_components, 'show_advanced', ui_config.get('show_advanced'))
            self._set_widget_value(form_components, 'auto_refresh', ui_config.get('auto_refresh'))
            self._set_widget_value(form_components, 'preview_enabled', ui_config.get('preview_enabled'))
            
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