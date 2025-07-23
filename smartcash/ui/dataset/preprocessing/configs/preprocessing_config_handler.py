"""
File: smartcash/ui/dataset/preprocessing/configs/preprocessing_config_handler.py
Description: Refactored preprocessing config handler - eliminated redundancies and overlaps.
"""

from typing import Dict, Any, Optional

from smartcash.ui.logger import get_module_logger
from smartcash.ui.dataset.preprocessing.configs.preprocessing_defaults import get_default_config
from smartcash.ui.dataset.preprocessing.constants import YOLO_PRESETS, CleanupTarget


class PreprocessingConfigHandler:
    """
    Streamlined preprocessing configuration handler.
    
    This refactored version eliminates redundancies and provides a clean,
    single-responsibility interface for configuration management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the preprocessing config handler with minimal setup."""
        self.logger = get_module_logger('smartcash.ui.dataset.preprocessing.configs')
        self._config = config.copy() if config else get_default_config()
        self._ui_components: Optional[Dict[str, Any]] = None
        self.logger.info("✅ Preprocessing config handler initialized")

    # Core Configuration Interface
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the internal configuration."""
        if not config:
            self._config = get_default_config()
        else:
            self._config = config.copy()
        self.logger.debug("Config updated")

    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration."""
        return get_default_config()

    # UI Integration Interface
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components for configuration extraction."""
        self._ui_components = ui_components
        self.logger.debug(f"UI components set: {list(ui_components.keys()) if ui_components else 'None'}")

    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI components."""
        if not self._ui_components:
            self.logger.warning("No UI components available")
            return self.get_current_config()

        config = self.get_current_config()
        prep_cfg = config.setdefault('preprocessing', {})
        norm_cfg = prep_cfg.setdefault('normalization', {})
        val_cfg = prep_cfg.setdefault('validation', {})

        # Extract values with safe fallbacks
        norm_cfg['preset'] = self._get_ui_value('resolution_dropdown', 'yolov5s')
        norm_cfg['method'] = self._get_ui_value('normalization_dropdown', 'minmax')
        norm_cfg['preserve_aspect_ratio'] = self._get_ui_value('preserve_aspect_checkbox', True)
        prep_cfg['target_splits'] = list(self._get_ui_value('target_splits_select', []))
        prep_cfg['batch_size'] = self._get_ui_value('batch_size_input', 32)
        val_cfg['enabled'] = self._get_ui_value('validation_checkbox', False)
        prep_cfg['move_invalid'] = self._get_ui_value('move_invalid_checkbox', False)
        prep_cfg['invalid_dir'] = self._get_ui_value('invalid_dir_input', 'data/invalid')
        prep_cfg['cleanup_target'] = self._get_ui_value('cleanup_target_dropdown', CleanupTarget.PREPROCESSED.value)
        prep_cfg['backup_enabled'] = self._get_ui_value('backup_checkbox', True)

        # Apply preset-specific settings
        if norm_cfg.get('preset') in YOLO_PRESETS:
            norm_cfg['target_size'] = YOLO_PRESETS[norm_cfg['preset']]['target_size']

        self._config = config
        return config

    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Update UI components from configuration."""
        if not self._ui_components:
            self.logger.warning("No UI components available to update")
            return

        config = config or self._config
        prep_cfg = config.get('preprocessing', {})
        norm_cfg = prep_cfg.get('normalization', {})
        val_cfg = prep_cfg.get('validation', {})

        # Update UI components
        updates = {
            'resolution_dropdown': norm_cfg.get('preset', 'yolov5s'),
            'normalization_dropdown': norm_cfg.get('method', 'minmax'),
            'preserve_aspect_checkbox': norm_cfg.get('preserve_aspect_ratio', True),
            'target_splits_select': tuple(prep_cfg.get('target_splits', [])),
            'batch_size_input': prep_cfg.get('batch_size', 32),
            'validation_checkbox': val_cfg.get('enabled', False),
            'move_invalid_checkbox': prep_cfg.get('move_invalid', False),
            'invalid_dir_input': prep_cfg.get('invalid_dir', 'data/invalid'),
            'cleanup_target_dropdown': prep_cfg.get('cleanup_target', CleanupTarget.PREPROCESSED.value),
            'backup_checkbox': prep_cfg.get('backup_enabled', True)
        }

        for component_key, value in updates.items():
            self._set_ui_value(component_key, value)

    # Configuration Operations
    def save_config(self) -> Dict[str, Any]:
        """Save the current configuration."""
        try:
            # Extract from UI if available, otherwise use current config
            if self._ui_components:
                self.extract_config_from_ui()
            
            config = self.get_current_config()
            self.logger.info(f"✅ Configuration saved with {len(config)} keys")
            return {
                'success': True,
                'message': f"Configuration saved with {len(config)} keys",
                'config': config
            }
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'config': self._config
            }

    def reset_config(self) -> Dict[str, Any]:
        """Reset configuration to defaults."""
        try:
            self._config = self.get_default_config()
            
            # Update UI if available
            if self._ui_components:
                self.update_ui_from_config()
                
            self.logger.info("✅ Configuration reset to defaults")
            return {
                'success': True,
                'message': "Configuration reset to defaults",
                'config': self._config
            }
        except Exception as e:
            error_msg = f"Failed to reset configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'config': self._config
            }

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate the configuration."""
        try:
            config = config or self._config
            errors = []
            warnings = []

            # Check preprocessing section
            preprocessing = config.get('preprocessing', {})
            if not preprocessing:
                errors.append("Missing preprocessing configuration section")
            else:
                # Validate target splits
                target_splits = preprocessing.get('target_splits', [])
                if not target_splits:
                    warnings.append("No target splits specified")

                # Validate batch size
                batch_size = preprocessing.get('batch_size', 0)
                if batch_size <= 0:
                    errors.append("Batch size must be greater than 0")

                # Check normalization
                normalization = preprocessing.get('normalization', {})
                if not normalization:
                    warnings.append("No normalization configuration found")

            # Check data section
            data = config.get('data', {})
            if not data:
                errors.append("Missing data configuration section")

            is_valid = len(errors) == 0
            return {
                'valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'config': config
            }
        except Exception as e:
            error_msg = f"Failed to validate configuration: {str(e)}"
            self.logger.error(error_msg)
            return {
                'valid': False,
                'errors': [error_msg],
                'warnings': [],
                'config': config or {}
            }

    # Private Helper Methods
    def _get_ui_value(self, component_key: str, default=None) -> Any:
        """Get a UI component value with error handling."""
        try:
            if self._ui_components and component_key in self._ui_components:
                component = self._ui_components[component_key]
                if hasattr(component, 'value'):
                    return component.value
                return component
            return default
        except Exception as e:
            self.logger.warning(f"Failed to get UI value for '{component_key}': {e}")
            return default

    def _set_ui_value(self, component_key: str, value: Any) -> None:
        """Set a UI component value with error handling."""
        try:
            if self._ui_components and component_key in self._ui_components:
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