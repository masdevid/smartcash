"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Enhanced preprocessing initializer dengan API integration dan dialog support
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """Enhanced preprocessing initializer dengan API integration, progress tracking, dan dialog support"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan API integration"""
        try:
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
                'progress_tracking_enabled': True
            })
            
            self.logger.info("✅ UI components created dengan API integration")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            raise
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan API integration dan progress tracking"""
        try:
            # Setup handlers dengan API integration
            result = setup_preprocessing_handlers(ui_components, config, env)
            
            # Load config dan update UI
            self._load_and_update_ui(ui_components)
            
            self.logger.info("✅ Handlers setup dengan API integration")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {str(e)}")
            return ui_components
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """Load config dari file dan update UI components"""
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                self.logger.warning("⚠️ Config handler tidak tersedia")
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
                self.logger.info("📂 Config loaded dan UI updated")
            else:
                self.logger.warning("⚠️ Config kosong, menggunakan defaults")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API compatibility"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading defaults: {str(e)}")
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
        """Critical components yang harus ada"""
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'confirmation_area'  # Dialog area juga critical
        ]

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs):
    """
    Factory function untuk preprocessing UI dengan API integration.
    
    Args:
        env: Environment info (opsional)
        config: Initial config (opsional)
        **kwargs: Additional parameters
        
    Returns:
        UI components dictionary dengan API integration
    """
    return _preprocessing_initializer.initialize(env=env, config=config, **kwargs)