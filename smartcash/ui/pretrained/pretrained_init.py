# File: smartcash/ui/pretrained/pretrained_initializer.py
"""
File: smartcash/ui/pretrained/pretrained_initializer.py
Deskripsi: Complete initializer untuk pretrained module dengan CommonInitializer patterns
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.pretrained_handlers import setup_pretrained_handlers
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedInitializer(CommonInitializer):
    """🚀 Pretrained module initializer dengan complete workflow"""
    
    def __init__(self):
        super().__init__(
            module_name='pretrained_models',
            config_filename='pretrained_config.yaml',
            config_handler_class=PretrainedConfigHandler
        )
    
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan validation"""
        from smartcash.ui.pretrained.components.ui_components import create_pretrained_ui
        
        ui_components = create_pretrained_ui(config)
        
        # Validate required components
        required_widgets = [
            'download_sync_button', 'save_button', 'reset_button',
            'status_panel', 'log_output'
        ]
        missing = [w for w in required_widgets if w not in ui_components]
        if missing:
            logger.warning(f"⚠️ Missing widgets: {missing}")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan environment check"""
        # Environment-specific setup
        if env == 'colab':
            ui_components['drive_enabled'] = True
            logger.info("🔧 Colab environment detected - drive sync enabled")
        
        return setup_pretrained_handlers(ui_components, config)
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs):
        """Post-init dengan auto-load config"""
        try:
            # Auto-load config ke UI
            config_handler = ui_components.get('config_handler')
            if config_handler and config:
                config_handler.update_ui(ui_components, config)
                self._log_post_init(ui_components, "✅ Config loaded", "success")
            
            # Setup auto-save callbacks
            self._setup_auto_save_callbacks(ui_components)
            
            # Initial status
            self._log_post_init(ui_components, "🎯 Pretrained models ready", "info")
            
        except Exception as e:
            logger.warning(f"⚠️ Post-init warning: {str(e)}")
    
    def _setup_auto_save_callbacks(self, ui_components: Dict[str, Any]):
        """Setup auto-save untuk form changes"""
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler or not hasattr(config_handler, 'config_mapping'):
                return
            
            def auto_save_callback(change=None):
                """Auto-save ketika ada perubahan"""
                try:
                    if hasattr(config_handler, 'set_ui_components'):
                        config_handler.set_ui_components(ui_components)
                    # Auto-save bisa diimplementasi di sini
                except Exception as e:
                    logger.debug(f"Auto-save error: {str(e)}")
            
            # Attach callbacks ke widgets
            for widget_key in config_handler.config_mapping.values():
                if widget := ui_components.get(widget_key):
                    if hasattr(widget, 'observe'):
                        widget.observe(auto_save_callback, names='value')
                        
        except Exception as e:
            logger.debug(f"Auto-save setup error: {str(e)}")
    
    def _log_post_init(self, ui_components: Dict[str, Any], message: str, level: str = "info"):
        """Helper untuk post-init logging"""
        if status_panel := ui_components.get('status_panel'):
            status_panel.value = message
        
        if log_output := ui_components.get('log_output'):
            with log_output:
                emoji_map = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
                print(f"{emoji_map.get(level, 'ℹ️')} {message}")

# Entry point function
def initialize_pretrained_ui(config: Optional[Dict[str, Any]] = None, env: Optional[str] = None) -> Any:
    """
    🚀 Entry point untuk pretrained module initialization
    
    Args:
        config: Optional config dictionary
        env: Environment type ('colab', 'local', etc.)
        
    Returns:
        UI widget atau container
    """
    try:
        initializer = PretrainedInitializer()
        return initializer.initialize(config=config, env=env)
        
    except Exception as e:
        logger.error(f"❌ Error initialize pretrained UI: {str(e)}")
        # Fallback UI
        import ipywidgets as widgets
        return widgets.HTML(f"<div style='color: red;'>❌ Error: {str(e)}</div>")

# Convenience functions
def create_pretrained_ui_quick(pretrained_type: str = 'yolov5s', models_dir: str = '/content/models') -> Any:
    """🔧 Quick setup dengan basic config"""
    quick_config = {
        'pretrained_models': {
            'pretrained_type': pretrained_type,
            'models_dir': models_dir,
            'auto_download': True,
            'sync_drive': False
        }
    }
    return initialize_pretrained_ui(config=quick_config)

def create_pretrained_ui_colab(drive_sync: bool = True) -> Any:
    """☁️ Colab-optimized setup"""
    colab_config = {
        'pretrained_models': {
            'models_dir': '/content/models',
            'drive_models_dir': '/content/drive/MyDrive/smartcash/models',
            'pretrained_type': 'yolov5s',
            'auto_download': True,
            'sync_drive': drive_sync
        }
    }
    return initialize_pretrained_ui(config=colab_config, env='colab')