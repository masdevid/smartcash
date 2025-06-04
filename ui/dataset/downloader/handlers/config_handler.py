"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler dengan proper YAML structure preservation dan one-liner style
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG, get_default_downloader_config
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.common.logger import get_logger

class DownloaderConfigHandler(ConfigHandler):
    """Fixed downloader config handler dengan YAML structure preservation."""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name)
        self.parent_module = parent_module
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan YAML structure preservation."""
        try:
            # Base config structure dari YAML files
            config = get_default_downloader_config()
            
            # Extract UI values dengan one-liner style
            roboflow_updates = {k: self._get_ui_value(ui_components, f"{k}_field", v) 
                              for k, v in [
                                  ('workspace', config['roboflow']['workspace']),
                                  ('project', config['roboflow']['project']),
                                  ('version', config['roboflow']['version']),
                                  ('api_key', config['roboflow']['api_key'])
                              ]}
            
            local_updates = {k: self._get_ui_value(ui_components, f"{k}_{'checkbox' if isinstance(v, bool) else 'field'}", v)
                           for k, v in [
                               ('output_dir', config['local']['output_dir']),
                               ('backup_dir', config['local']['backup_dir']),
                               ('organize_dataset', config['local']['organize_dataset']),
                               ('backup_enabled', config['local']['backup_enabled'])
                           ]}
            
            # Update config dengan preserved structure
            config['roboflow'].update(roboflow_updates)
            config['local'].update(local_updates)
            
            # Add format dari dropdown
            config['roboflow']['format'] = self._get_ui_value(ui_components, 'format_dropdown', 'yolov5pytorch')
            
            self._current_config = config
            self.logger.debug(f"📋 Config extracted with YAML structure preserved")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Extract config error: {str(e)}")
            return get_default_downloader_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan preserved YAML structure."""
        try:
            # Update roboflow section
            roboflow = config.get('roboflow', {})
            [self._set_ui_value(ui_components, f"{k}_field", v) for k, v in roboflow.items() if k != 'download_options']
            
            # Update local section
            local = config.get('local', {})
            [self._set_ui_value(ui_components, f"{k}_{'checkbox' if k in ['organize_dataset', 'backup_enabled'] else 'field'}", v) 
             for k, v in local.items()]
            
            # Update format dropdown
            self._set_ui_value(ui_components, 'format_dropdown', roboflow.get('format', 'yolov5pytorch'))
            
            self._current_config = config
            self.logger.debug("✅ UI updated with YAML structure")
            
        except Exception as e:
            self.logger.error(f"❌ Update UI error: {str(e)}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan YAML structure."""
        return get_default_downloader_config()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config."""
        return self._current_config.copy() if self._current_config else self.get_default_config()
    
    def _get_ui_value(self, ui_components: Dict[str, Any], key: str, default: Any) -> Any:
        """Get UI value dengan type-safe extraction."""
        return try_operation_safe(
            lambda: getattr(ui_components.get(key), 'value', default),
            fallback_value=default
        )
    
    def _set_ui_value(self, ui_components: Dict[str, Any], key: str, value: Any) -> None:
        """Set UI value dengan type-safe setting."""
        widget = ui_components.get(key)
        widget and hasattr(widget, 'value') and try_operation_safe(
            lambda: setattr(widget, 'value', value)
        )