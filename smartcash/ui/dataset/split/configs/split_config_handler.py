"""
File: smartcash/ui/dataset/split/configs/split_config_handler.py
Description: Configuration handler for split module following UIModule pattern
"""

from typing import Dict, Any, Optional


class SplitConfigHandler:
    """
    Split configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """Initialize the split config handler."""
        from smartcash.ui.logger import get_module_logger
        self.logger = logger or get_module_logger('smartcash.ui.dataset.split.configs')
        self.module_name = 'split'
        self.parent_module = 'dataset'
        from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
        self._config = config or get_default_split_config()
        self.config = self._config  # Backwards compatibility
        self.logger.info("✅ Split config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    # --- Abstract Method Implementations ---

    def get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration for the module."""
        from smartcash.ui.dataset.split.configs.split_defaults import get_default_split_config
        return get_default_split_config()

    def create_config_handler(self, config: Dict[str, Any]) -> 'SplitConfigHandler':
        """Returns self as the config handler instance."""
        return self

    # --- UI Integration ---

    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Set UI components for configuration extraction.
        
        Args:
            ui_components: Dictionary of UI components
        """
        self._ui_components = ui_components
        self.logger.debug(f"UI components set: {list(ui_components.keys()) if ui_components else 'None'}")

    def get_ui_value(self, component_key: str, default=None) -> Any:
        """
        Get a UI component value with error handling.
        
        Args:
            component_key: Key of the component in ui_components
            default: Default value if component not found or error occurs
            
        Returns:
            Component value or default
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components and component_key in self._ui_components:
                component = self._ui_components[component_key]
                if hasattr(component, 'value'):
                    return component.value
                return component
            return default
        except Exception as e:
            self.logger.warning(f"Failed to get UI value for '{component_key}': {e}")
            return default

    def set_ui_value(self, component_key: str, value: Any) -> None:
        """
        Set a UI component value with error handling.
        
        Args:
            component_key: Key of the component in ui_components
            value: Value to set for the component
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components and component_key in self._ui_components:
                component = self._ui_components[component_key]
                if hasattr(component, 'value'):
                    component.value = value
                    self.logger.debug(f"Set UI value for '{component_key}': {value}")
                else:
                    self.logger.warning(f"Component '{component_key}' does not have 'value' attribute")
            else:
                self.logger.warning(f"UI component '{component_key}' not found")
        except Exception as e:
            self.logger.warning(f"Failed to set UI value for '{component_key}': {e}")

    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extracts the current configuration from the UI components."""
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.logger.warning("No UI components available")
            return self.get_current_config()

        config = self.get_current_config()
        split_cfg = config.setdefault('split', {})
        ratios_cfg = split_cfg.setdefault('ratios', {})

        ratios_cfg['train'] = self.get_ui_value('train_ratio_input', default=0.7)
        ratios_cfg['val'] = self.get_ui_value('val_ratio_input', default=0.15)
        ratios_cfg['test'] = self.get_ui_value('test_ratio_input', default=0.15)
        split_cfg['input_dir'] = self.get_ui_value('input_dir_input', default='data/train')
        split_cfg['output_dir'] = self.get_ui_value('output_dir_input', default='data/split')
        split_cfg['method'] = self.get_ui_value('split_method_dropdown', default='random')
        split_cfg['seed'] = self.get_ui_value('seed_input', default=42)
        split_cfg['shuffle'] = self.get_ui_value('shuffle_checkbox', default=True)
        split_cfg['preserve_structure'] = self.get_ui_value('preserve_structure_checkbox', default=True)
        split_cfg['overwrite'] = self.get_ui_value('overwrite_checkbox', default=False)
        split_cfg['backup'] = self.get_ui_value('backup_checkbox', default=True)
        split_cfg['show_advanced'] = self.get_ui_value('show_advanced_checkbox', default=False)

        return config

    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Updates the UI components with values from the configuration."""
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.logger.warning("No UI components available to update")
            return

        split_cfg = config.get('split', {})
        ratios_cfg = split_cfg.get('ratios', {})

        self.set_ui_value('train_ratio_input', ratios_cfg.get('train', 0.7))
        self.set_ui_value('val_ratio_input', ratios_cfg.get('val', 0.15))
        self.set_ui_value('test_ratio_input', ratios_cfg.get('test', 0.15))
        self.set_ui_value('input_dir_input', split_cfg.get('input_dir', 'data/train'))
        self.set_ui_value('output_dir_input', split_cfg.get('output_dir', 'data/split'))
        self.set_ui_value('split_method_dropdown', split_cfg.get('method', 'random'))
        self.set_ui_value('seed_input', split_cfg.get('seed', 42))
        self.set_ui_value('shuffle_checkbox', split_cfg.get('shuffle', True))
        self.set_ui_value('preserve_structure_checkbox', split_cfg.get('preserve_structure', True))
        self.set_ui_value('overwrite_checkbox', split_cfg.get('overwrite', False))
        self.set_ui_value('backup_checkbox', split_cfg.get('backup', True))
        self.set_ui_value('show_advanced_checkbox', split_cfg.get('show_advanced', False))

    def save_config(self) -> Dict[str, Any]:
        """
        Save the current configuration.
        
        Returns:
            Dict with operation result
        """
        try:
            current_config = self.get_current_config()
            self.logger.info(f"✅ Split configuration saved with {len(current_config)} keys")
            return {
                'success': True,
                'message': f"Split configuration saved with {len(current_config)} keys",
                'config': current_config
            }
        except Exception as e:
            error_msg = f"Failed to save split configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }

    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Dict with operation result
        """
        try:
            default_config = self.get_default_config()
            self._merged_config = default_config.copy()
            # Update UI if components are available
            if hasattr(self, '_ui_components') and self._ui_components:
                self.update_ui_from_config(self._merged_config)
            self.logger.info("✅ Split configuration reset to defaults")
            return {
                'success': True,
                'message': f"Split configuration reset to defaults",
                'config': self._merged_config
            }
        except Exception as e:
            error_msg = f"Failed to reset split configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dict with validation result
        """
        try:
            current_config = self.get_current_config()
            errors = []
            warnings = []
            split = current_config.get('split', {})
            ratios = split.get('ratios', {})
            # Check ratios sum
            ratios_sum = sum(ratios.values()) if ratios else 0
            if not (0.999 <= ratios_sum <= 1.001):
                errors.append(f"Split ratios must sum to 1.0 (got {ratios_sum:.3f})")
            # Check each ratio
            for k, v in ratios.items():
                if v < 0.0 or v > 1.0:
                    errors.append(f"Ratio '{k}' must be between 0.0 and 1.0 (got {v})")
            # Check input/output dirs
            if not split.get('input_dir'):
                errors.append("Input directory is required")
            if not split.get('output_dir'):
                errors.append("Output directory is required")
            # Check method
            if split.get('method') not in ['random', 'stratified']:
                warnings.append(f"Unknown split method: {split.get('method')}")
            is_valid = len(errors) == 0
            if is_valid:
                self.logger.info("✅ Split configuration validation passed")
            else:
                self.logger.error(f"❌ Split configuration validation failed: {', '.join(errors)}")
            return {
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'config': current_config
            }
        except Exception as e:
            error_msg = f"Failed to validate split configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'valid': False,
                'errors': [error_msg],
                'warnings': [],
                'config': getattr(self, '_merged_config', {})
            }