"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Refactored initializer inheriting from CommonInitializer with service layer integration
"""

from typing import Dict, Any, List, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE

# Import components
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui

# Import enhanced SRP handlers dengan service integration
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
from smartcash.ui.dataset.preprocessing.handlers.dataset_checker import setup_dataset_checker
from smartcash.ui.dataset.preprocessing.handlers.cleanup_executor import setup_cleanup_executor  
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_executor import setup_preprocessing_executor
from smartcash.ui.dataset.preprocessing.handlers.progress_handlers import setup_progress_handlers

# Import service layer
from smartcash.ui.dataset.preprocessing.services.ui_preprocessing_service import create_ui_preprocessing_service


class PreprocessingInitializer(CommonInitializer):
    """
    Preprocessing UI initializer inheriting from CommonInitializer.
    Provides service layer integration and preprocessing-specific functionality.
    """
    
    def __init__(self):
        super().__init__(
            module_name='dataset_preprocessing',
            logger_namespace=PREPROCESSING_LOGGER_NAMESPACE
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create preprocessing UI components with service integration."""
        try:
            ui_components = create_preprocessing_main_ui(config)
            
            # Add essential metadata for preprocessing
            ui_components.update({
                'preprocessing_initialized': True,
                'service_integration_enabled': True,
                'config_manager': get_config_manager(),
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'preprocessed_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            })
            
            return ui_components
            
        except Exception as e:
            raise Exception(f"Enhanced UI component creation failed: {str(e)}")
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup preprocessing-specific handlers with service integration."""
        
        # Setup progress handlers FIRST (critical for service integration)
        ui_components = setup_progress_handlers(ui_components)
        
        # Setup service layer integration
        ui_service = create_ui_preprocessing_service(ui_components)
        ui_components['ui_service'] = ui_service
        
        # Setup handlers with service integration
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
        
        # Log warnings for failed handlers but continue
        if failed_handlers:
            ui_components['setup_warnings'] = failed_handlers
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Some handlers failed: {'; '.join(failed_handlers)}")
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get environment-aware default configuration for preprocessing."""
        try:
            # Environment-aware default config
            env_manager = get_environment_manager()
            paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
            
            return {
                'data': {'dir': paths['data_root']},
                'preprocessing': {
                    'img_size': [640, 640],
                    'normalize': True,
                    'normalization_method': 'minmax',
                    'num_workers': 4,
                    'split': 'all',
                    'preserve_aspect_ratio': True,
                    'output_dir': paths.get('preprocessed', 'data/preprocessed')
                }
            }
        except Exception:
            # Fallback to basic config
            return {
                'data': {'dir': 'data'},
                'preprocessing': {
                    'img_size': [640, 640], 
                    'normalize': True, 
                    'num_workers': 4, 
                    'split': 'all'
                }
            }
    
    def _get_critical_components(self) -> List[str]:
        """Get list of critical components that must exist for preprocessing."""
        return [
            'ui', 'preprocess_button', 'cleanup_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'ui_service', 'show_for_operation', 'update_progress', 
            'complete_operation', 'error_operation'
        ]
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation for service integration setup."""
        
        # Critical handler functions
        required_handlers = [
            'execute_preprocessing', 'execute_check', 'execute_cleanup',
            'save_config', 'reset_config'
        ]
        
        missing_handlers = []
        for handler in required_handlers:
            if handler not in ui_components:
                missing_handlers.append(f"Handler: {handler}")
        
        if missing_handlers:
            return {
                'valid': False,
                'message': f"Missing critical handlers: {', '.join(missing_handlers)}"
            }
        
        # Validate service layer functionality
        if 'ui_service' not in ui_components:
            return {
                'valid': False,
                'message': "UI service not initialized"
            }
        
        service_status = ui_components['ui_service'].get_service_status()
        if not any(service_status.values()):
            return {
                'valid': False,
                'message': "Service layer tidak properly initialized"
            }
        
        return {
            'valid': True,
            'message': 'Complete service integration setup berhasil',
            'service_status': service_status
        }
    
    def _get_merged_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced config merging with environment detection and path resolution."""
        try:
            # Get default config (environment-aware)
            merged_config = self._get_default_config()
            
            # Load saved config from config manager
            config_manager = get_config_manager()
            saved_config = {}
            try:
                saved_config = config_manager.get_config('preprocessing') or {}
            except Exception:
                pass
            
            # Merge saved config intelligently
            if saved_config and 'preprocessing' in saved_config:
                merged_config['preprocessing'].update(saved_config['preprocessing'])
            
            # Merge runtime config
            if config:
                if 'preprocessing' in config:
                    merged_config['preprocessing'].update(config['preprocessing'])
                if 'data' in config:
                    merged_config['data'].update(config['data'])
            
            return merged_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error merging config, using default: {str(e)}")
            return self._get_default_config()
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached UI components with new preprocessing config."""
        if not self._cached_components:
            return
        
        try:
            # Update config
            current_config = self._get_merged_config(new_config)
            self._cached_components['config'] = current_config
            
            # Update UI service with new config
            if 'ui_service' in self._cached_components:
                self._cached_components['ui_service'].cleanup_service_cache()
            
            # Apply config to UI
            from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
            config_extractor = get_config_extractor(self._cached_components)
            config_extractor.apply_config_to_ui(current_config)
            
        except Exception as e:
            logger = self._cached_components.get('logger', self.logger)
            logger.warning(f"⚠️ Config refresh error: {str(e)}")
    
    def _setup_log_suppression(self) -> None:
        """Enhanced log suppression for preprocessing service layer."""
        # Call parent method first
        super()._setup_log_suppression()
        
        # Add preprocessing-specific suppressions
        preprocessing_loggers = [
            'smartcash.dataset.preprocessor', 
            'smartcash.dataset.preprocessor.core',
            'smartcash.dataset.preprocessor.processors', 
            'smartcash.dataset.preprocessor.operations',
            'smartcash.dataset.services',
            'smartcash.dataset.utils', 
            'smartcash.common.threadpools',
            'concurrent.futures', 
            'PIL', 'cv2', 'numpy', 
            'matplotlib', 'tqdm'
        ]
        
        import logging
        for logger_name in preprocessing_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
            except Exception:
                pass
    
    def _create_error_fallback_ui(self, error_message: str):
        """Create preprocessing-specific error fallback UI."""
        import ipywidgets as widgets
        
        error_html = f"""
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                    border-radius: 8px; color: #856404; margin: 10px 0; max-width: 800px;">
            <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi Preprocessing UI</h4>
            <div style="margin: 15px 0;">
                <strong>Error Detail:</strong><br>
                <code style="background: #f8f9fa; padding: 5px; border-radius: 3px; font-size: 12px;">
                    {error_message}
                </code>
            </div>
            <div style="margin: 15px 0;">
                <strong>🔧 Solusi yang Bisa Dicoba:</strong>
                <ol style="margin: 10px 0; padding-left: 20px;">
                    <li>Restart kernel Colab dan jalankan ulang cell</li>
                    <li>Clear output semua cell dan jalankan dari awal</li>
                    <li>Periksa koneksi internet dan Google Drive</li> 
                    <li>Pastikan dataset sudah di-download terlebih dahulu</li>
                    <li>Coba dengan parameter <code>force_refresh=True</code></li>
                </ol>
            </div>
            <div style="margin: 15px 0; padding: 10px; background: #e8f4fd; border-radius: 5px;">
                <strong>💡 Quick Fix:</strong> Jalankan <code>reset_preprocessing_module()</code> kemudian coba lagi
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 5px;">
                <strong>🚀 Enhanced Features:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li>Service layer integration dengan preprocessing factory</li>
                    <li>Multi-level progress tracking dengan tqdm compatibility</li>
                    <li>Optimized batch processing dengan ThreadPoolExecutor</li>
                    <li>Symlink-safe cleanup operations</li>
                </ul>
            </div>
        </div>
        """
        
        return widgets.HTML(error_html)
    
    def cleanup_service_cache(self) -> None:
        """Cleanup service cache for preprocessing module."""
        if self._cached_components and 'ui_service' in self._cached_components:
            try:
                self._cached_components['ui_service'].cleanup_service_cache()
            except Exception as e:
                self.logger.warning(f"⚠️ Service cache cleanup failed: {str(e)}")
    
    def reset_module(self) -> None:
        """Reset module with service cleanup."""
        self.cleanup_service_cache()
        super().reset_module()


# Global instance
_preprocessing_initializer = PreprocessingInitializer()

# Public API functions
def initialize_dataset_preprocessing_ui(env=None, config=None, force_refresh=False):
    """
    Initialize preprocessing UI using the CommonInitializer pattern.
    
    Args:
        env: Environment manager (optional)
        config: Konfigurasi preprocessing (optional)
        force_refresh: Force refresh UI components
        
    Returns:
        Widget UI preprocessing atau error fallback UI
    """
    return _preprocessing_initializer.initialize(
        env=env, 
        config=config, 
        force_refresh=force_refresh
    )

def reset_preprocessing_module():
    """Reset preprocessing module initialization."""
    _preprocessing_initializer.reset_module()

def get_preprocessing_status():
    """Get preprocessing module status for debugging."""
    return _preprocessing_initializer.get_module_status()

def get_cached_preprocessing_components():
    """Get cached preprocessing components if available."""
    return _preprocessing_initializer.get_cached_components()

# Alias untuk kompatibilitas
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui