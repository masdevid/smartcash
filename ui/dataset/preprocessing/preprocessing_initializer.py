"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Fixed initializer tanpa cache complexity dengan service layer integration
"""

from typing import Dict, Any, List
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PREPROCESSING_LOGGER_NAMESPACE]

# Import components
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui

# Import handlers
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
from smartcash.ui.dataset.preprocessing.handlers.dataset_checker import setup_dataset_checker
from smartcash.ui.dataset.preprocessing.handlers.cleanup_executor import setup_cleanup_executor  
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_executor import setup_preprocessing_executor
from smartcash.ui.dataset.preprocessing.handlers.progress_handlers import setup_progress_handlers

# Import service layer
from smartcash.ui.dataset.preprocessing.services.ui_preprocessing_service import create_ui_preprocessing_service


class PreprocessingInitializer(CommonInitializer):
    """Fixed preprocessing initializer tanpa cache complexity"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, PREPROCESSING_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create preprocessing UI components dengan service integration"""
        ui_components = create_preprocessing_main_ui(config)
        
        ui_components.update({
            'preprocessing_initialized': True,
            'service_integration_enabled': True,
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup preprocessing handlers dengan service integration"""
        # Setup progress handlers FIRST
        ui_components = setup_progress_handlers(ui_components)
        
        # Setup service layer
        ui_service = create_ui_preprocessing_service(ui_components)
        ui_components['ui_service'] = ui_service
        
        # Setup handlers
        handlers_setup = [
            ("Config Handler", lambda: setup_config_handlers(ui_components, config)),
            ("Dataset Checker", lambda: setup_dataset_checker(ui_components)),
            ("Cleanup Executor", lambda: setup_cleanup_executor(ui_components)),
            ("Preprocessing Executor", lambda: setup_preprocessing_executor(ui_components, env))
        ]
        
        failed_handlers = []
        for handler_name, setup_func in handlers_setup:
            try:
                ui_components = setup_func()
            except Exception as e:
                failed_handlers.append(f"{handler_name}: {str(e)}")
        
        if failed_handlers:
            ui_components['setup_warnings'] = failed_handlers
            logger = ui_components.get('logger')
            logger and logger.warning(f"⚠️ Some handlers failed: {'; '.join(failed_handlers)}")
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get environment-aware default configuration"""
        try:
            env_manager = get_environment_manager()
            paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
            
            return {
                'data': {'dir': paths['data_root']},
                'preprocessing': {
                    'img_size': [640, 640], 'normalize': True, 'normalization_method': 'minmax',
                    'num_workers': 4, 'split': 'all', 'preserve_aspect_ratio': True,
                    'output_dir': paths.get('preprocessed', 'data/preprocessed')
                }
            }
        except Exception:
            return {
                'data': {'dir': 'data'},
                'preprocessing': {'img_size': [640, 640], 'normalize': True, 'num_workers': 4, 'split': 'all'}
            }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk preprocessing"""
        return [
            'ui', 'preprocess_button', 'cleanup_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'ui_service', 'show_for_operation', 'update_progress', 
            'complete_operation', 'error_operation'
        ]
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validation untuk service integration"""
        required_handlers = ['execute_preprocessing', 'execute_check', 'execute_cleanup', 'save_config', 'reset_config']
        missing_handlers = [handler for handler in required_handlers if handler not in ui_components]
        
        if missing_handlers:
            return {'valid': False, 'message': f"Missing handlers: {', '.join(missing_handlers)}"}
        
        if 'ui_service' not in ui_components:
            return {'valid': False, 'message': "UI service not initialized"}
        
        service_status = ui_components['ui_service'].get_service_status()
        if not any(service_status.values()):
            return {'valid': False, 'message': "Service layer tidak properly initialized"}
        
        return {'valid': True, 'message': 'Service integration setup berhasil', 'service_status': service_status}
    
    def _setup_log_suppression(self) -> None:
        """Enhanced log suppression untuk preprocessing"""
        super()._setup_log_suppression()
        
        preprocessing_loggers = [
            'smartcash.dataset.preprocessor', 'smartcash.dataset.preprocessor.core',
            'smartcash.dataset.preprocessor.processors', 'smartcash.dataset.preprocessor.operations',
            'smartcash.dataset.services', 'smartcash.dataset.utils', 'smartcash.common.threadpools',
            'concurrent.futures', 'PIL', 'cv2', 'numpy', 'matplotlib', 'tqdm'
        ]
        
        import logging
        for logger_name in preprocessing_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
            except Exception:
                pass


# Global instance
_preprocessing_initializer = PreprocessingInitializer()

# Public API functions - no cache
initialize_dataset_preprocessing_ui = lambda env=None, config=None, force_refresh=False: _preprocessing_initializer.initialize(env=env, config=config, **({'force_refresh': True} if force_refresh else {}))
reset_preprocessing_module = lambda: setattr(_preprocessing_initializer, '_last_result', None)  # Simple reset
get_preprocessing_status = lambda: _preprocessing_initializer.get_module_status()

# Alias
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui