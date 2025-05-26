"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Download initializer menggunakan CommonInitializer base class
"""

from typing import Dict, Any, Optional, List

# Import base class
from smartcash.ui.utils.common_initializer import CommonInitializer

# Konstanta namespace
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DOWNLOAD_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.progress_handlers import setup_progress_handlers
from smartcash.ui.dataset.download.components import create_download_ui


class DownloadInitializer(CommonInitializer):
    """
    Dataset download UI initializer menggunakan CommonInitializer base class.
    """
    
    def __init__(self):
        super().__init__(
            module_name='dataset_download',
            logger_namespace=DOWNLOAD_LOGGER_NAMESPACE
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration untuk download module."""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022', 
            'version': '3',
            'validate_dataset': True,
            'organize_dataset': True
        }
    
    def _get_critical_components(self) -> List[str]:
        """Get list of critical component keys yang harus ada."""
        return ['ui', 'download_button', 'check_button']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Create UI components specific untuk download module.
        
        Args:
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of UI components
        """
        return create_download_ui(config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers specific untuk download module.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Updated UI components dictionary
        """
        setup_results = {
            'config_handlers': False,
            'progress_handlers': False, 
            'button_handlers': False
        }
        
        # Setup config handlers
        try:
            ui_components = setup_config_handlers(ui_components, config)
            setup_results['config_handlers'] = True
        except Exception as e:
            logger = ui_components.get('logger', self.logger)
            logger.warning(f"⚠️ Config handlers setup failed: {str(e)}")
        
        # Setup progress handlers
        try:
            ui_components = setup_progress_handlers(ui_components)
            setup_results['progress_handlers'] = True
        except Exception as e:
            logger = ui_components.get('logger', self.logger)
            logger.warning(f"⚠️ Progress handlers setup failed: {str(e)}")
        
        # Setup button handlers (most critical)
        try:
            ui_components = setup_button_handlers(ui_components, env)
            setup_results['button_handlers'] = True
        except Exception as e:
            logger = ui_components.get('logger', self.logger)
            logger.error(f"❌ Button handlers setup failed: {str(e)}")
        
        # Store setup results untuk debugging
        ui_components['_setup_results'] = setup_results
        
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation specific untuk download module."""
        
        # Button functionality validation
        button_keys = ['download_button', 'check_button', 'cleanup_button', 'reset_button', 'save_button']
        functional_buttons = []
        
        for button_key in button_keys:
            if (button_key in ui_components and 
                ui_components[button_key] is not None and
                hasattr(ui_components[button_key], 'on_click')):
                functional_buttons.append(button_key)
        
        # Minimal requirement: download dan check button harus functional
        if 'download_button' not in functional_buttons:
            return {
                'valid': False,
                'message': 'Download button tidak functional'
            }
        
        if 'check_button' not in functional_buttons:
            return {
                'valid': False,
                'message': 'Check button tidak functional'  
            }
        
        return {
            'valid': True,
            'functional_buttons': functional_buttons,
            'total_functional': len(functional_buttons)
        }
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached UI components dengan config baru."""
        if not self._cached_components:
            return
        
        try:
            # Update base config
            super()._update_cached_config(new_config)
            
            # Update field values jika component ada
            field_mapping = {
                'workspace': 'workspace',
                'project': 'project',
                'version': 'version',
                'output_dir': 'output_dir'
            }
            
            for config_key, ui_key in field_mapping.items():
                if (config_key in new_config and 
                    ui_key in self._cached_components and
                    hasattr(self._cached_components[ui_key], 'value')):
                    self._cached_components[ui_key].value = new_config[config_key]
                    
        except Exception as e:
            # Silent fail untuk config update
            self.logger.debug(f"Config update failed: {str(e)}")


# Global initializer instance
_download_initializer = None

def get_download_initializer() -> DownloadInitializer:
    """Get atau create download initializer instance."""
    global _download_initializer
    if _download_initializer is None:
        _download_initializer = DownloadInitializer()
    return _download_initializer


# Public API functions - backward compatibility
def initialize_dataset_download_ui(env=None, config=None, force_refresh=False) -> Any:
    """
    Initialize UI download dataset dengan CommonInitializer pattern.
    
    Args:
        env: Environment context
        config: Custom configuration
        force_refresh: Force refresh UI components
        
    Returns:
        UI widget untuk download dataset
    """
    initializer = get_download_initializer()
    return initializer.initialize(env=env, config=config, force_refresh=force_refresh)

