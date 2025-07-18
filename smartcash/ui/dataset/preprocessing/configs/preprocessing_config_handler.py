"""
File: smartcash/ui/dataset/preprocessing/configs/preprocessing_config_handler.py
Description: Mixin-based config handler for the preprocessing module.
"""

from typing import Dict, Any, Optional

from smartcash.ui.core.mixins.configuration_mixin import ConfigurationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.logger import get_module_logger
from smartcash.ui.dataset.preprocessing.configs.preprocessing_defaults import get_default_config
from smartcash.ui.dataset.preprocessing.constants import YOLO_PRESETS, CleanupTarget


class PreprocessingConfigHandler(LoggingMixin, ConfigurationMixin):
    """
    Manages preprocessing configuration using a pure mixin-based approach.
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Initializes the PreprocessingConfigHandler."""
        super().__init__(**kwargs)
        self.module_name = 'preprocessing'
        self.parent_module = 'dataset'
        self.logger = get_module_logger('smartcash.ui.dataset.preprocessing.configs.preprocessing_config_handler')
        self._default_config = default_config or get_default_config()
        self._initialize_config_handler()
        self.logger.info("Modul konfigurasi preprocessing siap")

    # --- Abstract Method Implementations ---

    def get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration for the module."""
        return self._default_config.copy()

    def create_config_handler(self, config: Dict[str, Any]) -> 'PreprocessingConfigHandler':
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

    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extracts the current configuration from the UI components."""
        if not self._ui_components:
            self.log("Tidak ada komponen UI yang tersedia", 'warning')
            return self.get_current_config()

        config = self.get_current_config()
        prep_cfg = config.setdefault('preprocessing', {})
        norm_cfg = prep_cfg.setdefault('normalization', {})
        val_cfg = prep_cfg.setdefault('validation', {})

        norm_cfg['preset'] = self.get_ui_value('resolution_dropdown')
        norm_cfg['method'] = self.get_ui_value('normalization_dropdown')
        norm_cfg['preserve_aspect_ratio'] = self.get_ui_value('preserve_aspect_checkbox', default=True)
        prep_cfg['target_splits'] = list(self.get_ui_value('target_splits_select', default=[]))
        prep_cfg['batch_size'] = self.get_ui_value('batch_size_input', default=32)
        val_cfg['enabled'] = self.get_ui_value('validation_checkbox', default=False)
        prep_cfg['move_invalid'] = self.get_ui_value('move_invalid_checkbox', default=False)
        prep_cfg['invalid_dir'] = self.get_ui_value('invalid_dir_input', default='data/invalid')
        prep_cfg['cleanup_target'] = self.get_ui_value('cleanup_target_dropdown')
        prep_cfg['backup_enabled'] = self.get_ui_value('backup_checkbox', default=True)

        if norm_cfg.get('preset') in YOLO_PRESETS:
            norm_cfg['target_size'] = YOLO_PRESETS[norm_cfg['preset']]['target_size']

        return config

    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Updates the UI components with values from the configuration."""
        if not self._ui_components:
            self.log("Tidak ada komponen UI yang tersedia untuk diperbarui", 'warning')
            return

        prep_cfg = config.get('preprocessing', {})
        norm_cfg = prep_cfg.get('normalization', {})
        val_cfg = prep_cfg.get('validation', {})

        self.set_ui_value('resolution_dropdown', norm_cfg.get('preset', 'yolov5s'))
        self.set_ui_value('normalization_dropdown', norm_cfg.get('method', 'minmax'))
        self.set_ui_value('preserve_aspect_checkbox', norm_cfg.get('preserve_aspect_ratio', True))
        self.set_ui_value('target_splits_select', tuple(prep_cfg.get('target_splits', [])))
        self.set_ui_value('batch_size_input', prep_cfg.get('batch_size', 32))
        self.set_ui_value('validation_checkbox', val_cfg.get('enabled', False))
        self.set_ui_value('move_invalid_checkbox', prep_cfg.get('move_invalid', False))
        self.set_ui_value('invalid_dir_input', prep_cfg.get('invalid_dir', 'data/invalid'))
        self.set_ui_value('cleanup_target_dropdown', prep_cfg.get('cleanup_target', CleanupTarget.PREPROCESSED.value))
        self.set_ui_value('backup_checkbox', prep_cfg.get('backup_enabled', True))
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save the current configuration.
        
        Returns:
            Dict with operation result
        """
        try:
            current_config = self.get_current_config()
            
            self.log_with_status(
                message=f"Preprocessing configuration saved with {len(current_config)} keys",
                status_message="Konfigurasi preprocessing berhasil disimpan",
                log_level='info',
                status_level='success'
            )
            
            return {
                'success': True,
                'message': 'Konfigurasi preprocessing berhasil disimpan',
                'config': current_config
            }
            
        except Exception as e:
            error_msg = f"Gagal menyimpan konfigurasi preprocessing: {str(e)}"
            self.log_with_status(
                message=error_msg,
                status_message=error_msg,
                log_level='error',
                status_level='error'
            )
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
            
            self.log_with_status(
                message="Preprocessing configuration reset to defaults",
                status_message="Konfigurasi preprocessing direset ke pengaturan awal",
                log_level='info',
                status_level='success'
            )
            
            return {
                'success': True,
                'message': 'Konfigurasi preprocessing berhasil direset',
                'config': self._merged_config
            }
            
        except Exception as e:
            error_msg = f"Gagal mereset konfigurasi preprocessing: {str(e)}"
            self.log_with_status(
                message=error_msg,
                status_message=error_msg,
                log_level='error',
                status_level='error'
            )
            return {
                'success': False,
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }
