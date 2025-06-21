# File: smartcash/ui/pretrained/pretrained_initializer.py
"""
File: smartcash/ui/pretrained/pretrained_initializer.py
Deskripsi: Complete initializer untuk pretrained module dengan CommonInitializer patterns - Fixed abstract methods
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedInitializer(CommonInitializer):
    """🚀 Pretrained module initializer dengan complete workflow - Fixed abstract methods"""
    
    def __init__(self):
        super().__init__(
            module_name='pretrained_models',
            config_handler_class=PretrainedConfigHandler
        )
    
    # FIXED: Abstract method implementation - nama method sesuai parent class
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan validation - Fixed method name"""
        from smartcash.ui.pretrained.components.ui_components import create_pretrained_ui_components
        
        ui_components = create_pretrained_ui_components(config)
        
        # Validate required components
        required_widgets = [
            'download_sync_button', 'save_button', 'reset_button',
            'status_panel', 'log_output'
        ]
        missing = [w for w in required_widgets if w not in ui_components]
        if missing:
            logger.warning(f"⚠️ Missing widgets: {missing}")
        
        return ui_components
    
    # FIXED: Abstract method implementation
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup pretrained handlers dengan environment check"""
        from smartcash.ui.pretrained.handlers.pretrained_handlers import setup_pretrained_handlers
        
        # Environment-specific setup
        if env == 'colab':
            ui_components['drive_enabled'] = True
            logger.info("🌍 Google Drive enabled for Colab environment")
        
        return setup_pretrained_handlers(ui_components, config, env=env, **kwargs)
    
    # FIXED: Abstract method implementation  
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pretrained config dari defaults module"""
        from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
        return get_default_pretrained_config()
    
    # FIXED: Abstract method implementation
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada untuk pretrained module"""
        return [
            'ui', 'download_sync_button', 'save_button', 'reset_button',
            'status_panel', 'log_output', 'config_handler'
        ]
    
    # Post-initialization hook untuk auto-load models
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs):
        """Post-init hook untuk pretrained models dengan auto-sync check"""
        super()._post_initialization_hook(ui_components, config, env, **kwargs)
        
        # Auto-check model availability jika enabled
        if config.get('models', {}).get('download', {}).get('auto_download', False):
            try:
                from smartcash.ui.pretrained.handlers.model_checker import check_model_availability
                available_models = check_model_availability(config)
                ui_components['available_models'] = available_models
                logger.info(f"🎯 Available models: {len(available_models)} found")
            except Exception as e:
                logger.warning(f"⚠️ Model availability check failed: {str(e)}")

# Global instance
_pretrained_initializer = PretrainedInitializer()

def initialize_pretrained_ui(env=None, config=None, **kwargs):
    """
    Factory function untuk pretrained UI dengan model integration.
    
    Args:
        env: Environment info (opsional)  
        config: Initial config (opsional)
        **kwargs: Additional parameters
        
    Returns:
        UI components dictionary dengan pretrained model integration
    """
    return _pretrained_initializer.initialize(env=env, config=config, **kwargs)