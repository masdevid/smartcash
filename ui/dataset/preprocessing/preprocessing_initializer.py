"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Enhanced preprocessing initializer dengan API integration dan dialog support

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers
from smartcash.ui.handlers.error_handler import create_error_response

class PreprocessingInitializer(CommonInitializer):
    """Enhanced preprocessing initializer dengan API integration, progress tracking, dan dialog support

    Provides a structured approach to initializing the dataset preprocessing module with
    proper error handling, logging, and UI component management. Follows the same
    initialization flow as CommonInitializer with additional preprocessing-specific
    functionality.
    """

    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler
        )

    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan API integration

        Args:
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info(" Membuat komponen UI preprocessing")
            ui_components = create_preprocessing_main_ui(config)

            # Validate critical components
            missing = [name for name in self._get_critical_components() if name not in ui_components]
            if missing:
                raise ValueError(f"Missing critical components: {', '.join(missing)}")

            # Enhanced metadata
            ui_components.update({
                'preprocessing_initialized': True,
                'module_name': 'preprocessing',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'api_integration_enabled': True,
                'dialog_components_loaded': True,
                'progress_tracking_enabled': True,
                'logger': self.logger  # Ensure logger is available
            })

            self.logger.debug(f"UI components created: {list(ui_components.keys())}")
            return ui_components

        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI preprocessing")

    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan API integration dan progress tracking

        Args:
            ui_components: Dictionary of UI components
            config: Loaded configuration
            env: Optional environment context
            **kwargs: Additional arguments

        Returns:
            Updated UI components with handlers
        """
        try:
            self.logger.info(" Menyiapkan handlers preprocessing")

            # Setup handlers dengan API integration
            handlers = setup_preprocessing_handlers(ui_components, config, env)

            # Update UI components with handlers
            if handlers:
                ui_components.update(handlers)
                self.logger.debug(f"Handlers setup complete: {list(handlers.keys()) if handlers else 'No handlers'}")

            # Load config dan update UI
            self._load_and_update_ui(ui_components)

            return ui_components

        except Exception as e:
            self.handle_error(f"Failed to setup module handlers: {str(e)}", exc_info=True)
            return ui_components  # Return original components to avoid breaking the UI

    def _load_and_update_ui(self, ui_components: Dict[str, Any]) -> None:
        """Load config dari file dan update UI components

        Args:
            ui_components: Dictionary of UI components
        """
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                self.logger.warning(" Config handler tidak tersedia")
                return

            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)

            # Load config dengan inheritance support
            loaded_config = config_handler.load_config()
            if loaded_config:
                # Update UI dengan loaded config
                config_handler.update_ui(ui_components, loaded_config)
                ui_components['config'] = loaded_config
                self.logger.debug("Config loaded dan UI updated")
            else:
                self.logger.warning(" Config kosong, menggunakan defaults")

        except Exception as e:
            self.handle_error(f"Failed to load and update UI: {str(e)}", exc_info=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API compatibility

        Returns:
            Default configuration dictionary
        """
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            default_config = get_default_preprocessing_config()
            self.logger.debug("Default config loaded successfully")
            return default_config
        except Exception as e:
            self.handle_error(f"Failed to get default config: {str(e)}", exc_info=True)
            # Return minimal working config to prevent crashes
            return {
                'preprocessing': {
                    'enabled': True,
                    'target_splits': ['train', 'valid'],
                    'normalization': {'enabled': True, 'method': 'minmax', 'target_size': [640, 640]}
                },
                'performance': {'batch_size': 32},
                'data': {'dir': 'data'}
            }

    def _get_critical_components(self) -> List[str]:
        """Get list of critical UI components that must exist

        Returns:
            List of critical component keys
        """
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'log_output', 'status_panel', 'progress_tracker'
        ]

    def pre_initialize_checks(self, **kwargs) -> None:
        """Perform pre-initialization checks

        Raises:
            Exception: If any pre-initialization check fails
        """
        # Check if we're in a supported environment
        try:
            import IPython
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dataset preprocessing requires IPython environment")

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk preprocessing UI dengan API integration

    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components with 'ui' as the main component
    """
    return _preprocessing_initializer.initialize(config=config, env=env, **kwargs)