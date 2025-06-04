"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Fixed download initializer tanpa cache complexity
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DOWNLOAD_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.progress_handlers import setup_progress_handlers
from smartcash.ui.dataset.download.components import create_download_ui


class DownloadInitializer(CommonInitializer):
    """Fixed download initializer tanpa cache complexity"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, DOWNLOAD_LOGGER_NAMESPACE)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration untuk download module"""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022', 
            'version': '3',
            'validate_dataset': True,
            'organize_dataset': True
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical component keys yang harus ada"""
        return ['ui', 'download_button', 'check_button']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk download module"""
        return create_download_ui(config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk download module"""
        setup_results = {'config_handlers': False, 'progress_handlers': False, 'button_handlers': False}
        
        # Setup handlers dengan error handling
        handlers = [
            ('config_handlers', lambda: setup_config_handlers(ui_components, config)),
            ('progress_handlers', lambda: setup_progress_handlers(ui_components)),
            ('button_handlers', lambda: setup_button_handlers(ui_components, env))
        ]
        
        for key, setup_func in handlers:
            try:
                ui_components = setup_func()
                setup_results[key] = True
            except Exception as e:
                logger = ui_components.get('logger', self.logger)
                level = 'error' if key == 'button_handlers' else 'warning'
                getattr(logger, level)(f"{'❌' if level == 'error' else '⚠️'} {key.title().replace('_', ' ')} setup failed: {str(e)}")
        
        ui_components['_setup_results'] = setup_results
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validation untuk download module"""
        button_keys = ['download_button', 'check_button', 'cleanup_button', 'reset_button', 'save_button']
        functional_buttons = [key for key in button_keys if ui_components.get(key) and hasattr(ui_components[key], 'on_click')]
        
        # Check minimal requirements
        required_buttons = ['download_button', 'check_button']
        missing_required = [button for button in required_buttons if button not in functional_buttons]
        
        if missing_required:
            return {'valid': False, 'message': f'{missing_required[0].replace("_", " ").title()} tidak functional'}
        
        return {'valid': True, 'functional_buttons': functional_buttons, 'total_functional': len(functional_buttons)}


# Global instance - simplified
_download_initializer = DownloadInitializer()

# Public API - no cache
initialize_dataset_download_ui = lambda env=None, config=None, force_refresh=False: _download_initializer.initialize(env=env, config=config)