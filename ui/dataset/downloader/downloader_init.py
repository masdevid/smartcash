"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Improved downloader initializer dengan auto API key detection dan proper handlers
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.fallback_utils import create_fallback_ui, try_operation_safe, show_status_safe
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge, get_logger
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES, get_namespace_color
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config, get_default_api_key
from smartcash.common.environment import get_environment_manager

class DownloaderConfigHandler(BaseConfigHandler):
    """Config handler dengan auto API key detection dan YAML structure preservation."""
    
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
                'format': 'yolov5pytorch'  # Fixed format
            })
            
            # Update local section - organize_dataset always TRUE
            config['local'].update({
                'output_dir': self._get_safe_value(ui_components, 'output_dir_field', ''),
                'backup_dir': self._get_safe_value(ui_components, 'backup_dir_field', ''),
                'organize_dataset': True,  # Always TRUE
                'backup_enabled': self._get_safe_value(ui_components, 'backup_checkbox', False)
            })
            
            return config
        except Exception as e:
            self.logger.error(f"❌ Extract config error: {str(e)}")
            return get_default_downloader_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan YAML structure dan auto API key detection."""
        try:
            roboflow = config.get('roboflow', {})
            local = config.get('local', {})
            
            # Auto-detect API key dari Colab secrets jika belum ada
            api_key = roboflow.get('api_key', '') or get_default_api_key()
            
            # Update fields dengan safe setting
            field_updates = [
                ('workspace_field', roboflow.get('workspace', '')),
                ('project_field', roboflow.get('project', '')),
                ('version_field', roboflow.get('version', '1')),
                ('api_key_field', api_key),
                ('output_dir_field', local.get('output_dir', '')),
                ('backup_dir_field', local.get('backup_dir', '')),
                ('backup_checkbox', local.get('backup_enabled', False))
            ]
            
            [self._set_safe_value(ui_components, field, value) for field, value in field_updates]
            
            # Update status jika API key terdeteksi dari Colab secrets
            if api_key and not roboflow.get('api_key'):
                show_status_safe("🔑 API key terdeteksi dari Colab secrets", 'success', ui_components)
                
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
    """Fixed downloader initializer dengan proper action handlers."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan auto API key detection."""
        try:
            from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui
            
            # Auto-detect API key dari Colab secrets sebelum create UI
            if not config.get('roboflow', {}).get('api_key'):
                api_key = get_default_api_key()
                if api_key:
                    config.setdefault('roboflow', {})['api_key'] = api_key
            
            # Add environment manager
            env_manager = get_environment_manager()
            
            ui_components = create_downloader_ui(config, env_manager)
            ui_components.update({
                'downloader_initialized': True,
                'config': config,
                'module_name': 'downloader',
                'env_manager': env_manager
            })
            
            # Add progress tracking components
            progress_components = self._create_progress_components()
            ui_components.update(progress_components)
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            return create_fallback_ui(f"Error creating downloader UI: {str(e)}", 'downloader')
    
    def _create_progress_components(self) -> Dict[str, Any]:
        """Create progress tracking components."""
        try:
            from smartcash.ui.components.progress_tracking import create_progress_tracking_container
            return create_progress_tracking_container()
        except Exception:
            return {}
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup action handlers dengan proper confirmation dan logging."""
        try:
            # Import action handlers
            from smartcash.ui.dataset.downloader.handlers.action_handlers import setup_download_action_handlers
            
            # Setup all action handlers
            handler_result = setup_download_action_handlers(ui_components)
            
            if handler_result.get('status') == 'success':
                self.logger.info("✅ Action handlers configured successfully")
                return {'handlers_setup': True, 'handlers_count': handler_result.get('handlers_configured', 0)}
            else:
                self.logger.error(f"❌ Handler setup failed: {handler_result.get('message', 'Unknown error')}")
                return {'handlers_setup': False, 'error': handler_result.get('message')}
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up handlers: {str(e)}")
            return {'handlers_setup': False, 'error': str(e)}
    
    def _get_default_config(self) -> Dict[str, Any]:
        return get_default_downloader_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', 'download_button', 'workspace_field', 'project_field', 
            'status_panel', 'log_output', 'form_container'
        ]
    
    def _get_return_value(self, ui_components: Dict[str, Any]):
        return ui_components.get('ui', ui_components.get('main_container'))

# Global instance
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Initialize downloader UI dengan auto API key detection."""
    try:
        return _downloader_initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        logger = get_logger('downloader.init')
        logger.error(f"❌ Initialization failed: {str(e)}")
        return create_fallback_ui(f"Gagal inisialisasi downloader: {str(e)}", 'downloader')['ui']

def get_downloader_status() -> Dict[str, Any]:
    """Get downloader status."""
    return {'module': 'downloader', 'initialized': True, 'status': 'ready', 'config_available': True}

def setup_downloader(env=None, config=None, **kwargs):
    """Main entry point."""
    return initialize_downloader_ui(env=env, config=config, **kwargs)