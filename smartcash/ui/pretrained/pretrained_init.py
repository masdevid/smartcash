"""
File: smartcash/ui/pretrained/pretrained_init.py
Deskripsi: Enhanced pretrained initializer dengan API integration, progress tracking, dan dialog support
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.ui.pretrained.components.ui_components import create_pretrained_main_ui
from smartcash.ui.pretrained.handlers.pretrained_handlers import setup_pretrained_handlers

class PretrainedInitializer(CommonInitializer):
    """Enhanced pretrained initializer dengan API integration, progress tracking, dan dialog support"""
    
    def __init__(self):
        super().__init__(
            module_name='pretrained',
            config_handler_class=PretrainedConfigHandler,
            parent_module=None  # Top level module
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan API integration"""
        try:
            ui_components = create_pretrained_main_ui(config)
            
            # Validate critical components
            missing = [name for name in self._get_critical_components() if name not in ui_components]
            if missing:
                raise ValueError(f"Missing critical components: {', '.join(missing)}")
            
            # Enhanced metadata
            ui_components.update({
                'pretrained_initialized': True,
                'module_name': 'pretrained',
                'models_dir': config.get('pretrained_models', {}).get('models_dir', '/data/pretrained'),
                'api_integration_enabled': True,
                'dialog_confirmation_enabled': True
            })
            
            self.logger.info("✅ UI components berhasil dibuat")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            raise
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan progress tracker integration"""
        try:
            # Setup handlers dengan progress tracking support
            handlers_result = setup_pretrained_handlers(ui_components, config)
            
            # Update ui_components dengan handlers
            ui_components.update(handlers_result)
            
            self.logger.info("✅ Handlers setup berhasil dengan progress tracking")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error setup handlers: {str(e)}")
            return ui_components
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs):
        """Post initialization dengan config loading"""
        try:
            # Load existing config dan update UI
            if config:
                config_handler = ui_components.get('config_handler')
                if config_handler:
                    config_handler.update_ui(ui_components, config)
                    ui_components['config'] = config
                    self.logger.info("📂 Config loaded dan UI updated")
                else:
                    self.logger.warning("⚠️ Config handler not found")
            else:
                self.logger.warning("⚠️ Config kosong, menggunakan defaults")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API compatibility"""
        try:
            from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
            return get_default_pretrained_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading defaults: {str(e)}")
            return {
                'pretrained_models': {
                    'models_dir': '/data/pretrained',
                    'drive_models_dir': '/content/drive/MyDrive/SmartCash/pretrained',
                    'models': {
                        'yolov5s': {
                            'name': 'YOLOv5s',
                            'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                            'filename': 'yolov5s.pt',
                            'min_size_mb': 10,
                            'description': 'Object detection backbone'
                        },
                        'efficientnet_b4': {
                            'name': 'EfficientNet-B4',
                            'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                            'filename': 'efficientnet_b4_huggingface.bin',
                            'min_size_mb': 60,
                            'description': 'Feature extraction backbone'
                        }
                    }
                }
            }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return [
            'ui', 'download_sync_button', 'log_output', 'status_panel',
            'confirmation_area', 'progress_tracker', 'yolov5_url_input',
            'efficientnet_url_input'
        ]

# Global instance
_pretrained_initializer = PretrainedInitializer()

def initialize_pretrained_ui(env=None, config=None, **kwargs):
    """
    Factory function untuk pretrained UI dengan API integration.
    
    Args:
        env: Environment info (opsional)
        config: Initial config (opsional)
        **kwargs: Additional parameters
        
    Returns:
        UI components dictionary dengan API integration
    """
    return _pretrained_initializer.initialize(env=env, config=config, **kwargs)