"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Fixed downloader initializer yang langsung return UI widget
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.fallback_utils import create_fallback_ui, try_operation_safe, show_status_safe
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge, get_logger
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config

class DownloaderConfigHandler(BaseConfigHandler):
    """Simple config handler untuk downloader dengan YAML structure preservation."""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module=parent_module)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan YAML structure preservation."""
        try:
            config = get_default_downloader_config()
            
            # Update roboflow section
            config['roboflow'].update({
                'workspace': self._get_safe_value(ui_components, 'workspace_field', ''),
                'project': self._get_safe_value(ui_components, 'project_field', ''),
                'version': self._get_safe_value(ui_components, 'version_field', '1'),
                'api_key': self._get_safe_value(ui_components, 'api_key_field', ''),
                'format': self._get_safe_value(ui_components, 'format_dropdown', 'yolov5pytorch')
            })
            
            # Update local section
            config['local'].update({
                'output_dir': self._get_safe_value(ui_components, 'output_dir_field', ''),
                'backup_dir': self._get_safe_value(ui_components, 'backup_dir_field', ''),
                'organize_dataset': self._get_safe_value(ui_components, 'organize_dataset', True),
                'backup_enabled': self._get_safe_value(ui_components, 'backup_checkbox', False)
            })
            
            return config
        except Exception as e:
            self.logger.error(f"❌ Extract config error: {str(e)}")
            return get_default_downloader_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan YAML structure."""
        try:
            roboflow = config.get('roboflow', {})
            local = config.get('local', {})
            
            # Update fields dengan safe setting
            field_updates = [
                ('workspace_field', roboflow.get('workspace', '')),
                ('project_field', roboflow.get('project', '')),
                ('version_field', roboflow.get('version', '1')),
                ('api_key_field', roboflow.get('api_key', '')),
                ('output_dir_field', local.get('output_dir', '')),
                ('backup_dir_field', local.get('backup_dir', '')),
                ('organize_dataset', local.get('organize_dataset', True)),
                ('backup_checkbox', local.get('backup_enabled', False))
            ]
            
            [self._set_safe_value(ui_components, field, value) for field, value in field_updates]
            
        except Exception as e:
            self.logger.error(f"❌ Update UI error: {str(e)}")
    
    def get_default_config(self) -> Dict[str, Any]:
        return get_default_downloader_config()
    
    def _get_safe_value(self, ui_components: Dict[str, Any], key: str, default: Any) -> Any:
        """Get value safely dari UI component."""
        return try_operation_safe(
            lambda: getattr(ui_components.get(key), 'value', default),
            fallback_value=default
        )
    
    def _set_safe_value(self, ui_components: Dict[str, Any], key: str, value: Any) -> None:
        """Set value safely ke UI component."""
        widget = ui_components.get(key)
        widget and hasattr(widget, 'value') and try_operation_safe(
            lambda: setattr(widget, 'value', value)
        )

class DownloaderInitializer(CommonInitializer):
    """Fixed downloader initializer yang return UI widget langsung."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling."""
        try:
            # Import UI creation functions
            from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui
            
            # Create UI dengan config dan env
            ui_components = create_downloader_ui(config, env)
            
            # Add essential metadata
            ui_components.update({
                'downloader_initialized': True,
                'config': config,
                'module_name': 'downloader'
            })
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            return create_fallback_ui(f"Error creating downloader UI: {str(e)}", 'downloader')
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan error handling."""
        try:
            # Setup basic button handlers
            self._setup_basic_handlers(ui_components, config)
            return {'handlers_setup': True}
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {str(e)}")
            return {'handlers_setup': False, 'error': str(e)}
    
    def _setup_basic_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup basic button handlers yang simple."""
        logger = ui_components.get('logger', self.logger)
        
        # Download button handler
        def handle_download(b):
            show_status_safe("🚀 Download functionality akan segera tersedia", 'info', ui_components)
            logger.info("🚀 Download button clicked")
        
        # Check button handler  
        def handle_check(b):
            show_status_safe("🔍 Check functionality akan segera tersedia", 'info', ui_components)
            logger.info("🔍 Check button clicked")
        
        # Cleanup button handler
        def handle_cleanup(b):
            show_status_safe("🧹 Cleanup functionality akan segera tersedia", 'info', ui_components)
            logger.info("🧹 Cleanup button clicked")
        
        # Bind handlers dengan safe checking
        button_handlers = [
            ('download_button', handle_download),
            ('check_button', handle_check), 
            ('cleanup_button', handle_cleanup)
        ]
        
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config."""
        return get_default_downloader_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada."""
        return ['ui', 'download_button', 'workspace_field', 'project_field', 'status_panel']
    
    def _get_return_value(self, ui_components: Dict[str, Any]):
        """Return UI widget langsung, bukan dictionary."""
        return ui_components.get('ui', ui_components.get('main_container'))

# Global instance
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """
    Initialize downloader UI dan return widget langsung.
    
    Returns:
        UI widget yang bisa langsung di-display
    """
    try:
        return _downloader_initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        logger = get_logger('downloader.init')
        logger.error(f"❌ Initialization failed: {str(e)}")
        
        # Return fallback UI widget
        return create_fallback_ui(f"Gagal inisialisasi downloader: {str(e)}", 'downloader')['ui']

def get_downloader_status() -> Dict[str, Any]:
    """Get downloader status untuk debugging."""
    return {
        'module': 'downloader',
        'initialized': True,
        'status': 'ready',
        'config_available': True
    }

def setup_downloader(env=None, config=None, **kwargs):
    """Main entry point - alias untuk initialize_downloader_ui."""
    return initialize_downloader_ui(env=env, config=config, **kwargs)