"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler untuk downloader dengan implementasi lengkap abstract methods
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config

class DownloaderConfigHandler(ConfigHandler):
    """Fixed config handler untuk downloader dengan implementasi lengkap abstract methods"""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components - implementasi abstract method"""
        return self.extract_config_from_ui(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config - implementasi abstract method"""
        self.update_ui_from_config(ui_components, config)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari dataset_config.yaml dengan API key auto-detection"""
        config = get_default_downloader_config()
        return set_api_key_to_config(config, force_refresh=False)
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components dengan one-liner safe access"""
        return {
            'data': {
                'source': 'roboflow',
                'roboflow': {
                    'workspace': self._get_widget_value(ui_components, 'workspace_input', '').strip(),
                    'project': self._get_widget_value(ui_components, 'project_input', '').strip(),
                    'version': self._get_widget_value(ui_components, 'version_input', '').strip(),
                    'api_key': self._get_widget_value(ui_components, 'api_key_input', '').strip(),
                    'output_format': 'yolov5pytorch'
                },
                'file_naming': {
                    'uuid_format': True,
                    'naming_strategy': 'research_uuid',
                    'preserve_original': False
                }
            },
            'download': {
                'rename_files': True,
                'organize_dataset': True,
                'validate_download': self._get_widget_value(ui_components, 'validate_checkbox', True),
                'backup_existing': self._get_widget_value(ui_components, 'backup_checkbox', False),
                'retry_count': 3,
                'timeout': 30
            },
            'uuid_renaming': {
                'enabled': True,
                'backup_before_rename': self._get_widget_value(ui_components, 'backup_checkbox', False),
                'validate_consistency': True
            }
        }
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan safe widget updates"""
        roboflow = config.get('data', {}).get('roboflow', {})
        download = config.get('download', {})
        
        # Update UI widgets dengan safe operations
        widget_updates = [
            ('workspace_input', roboflow.get('workspace', '')),
            ('project_input', roboflow.get('project', '')),
            ('version_input', str(roboflow.get('version', ''))),
            ('api_key_input', roboflow.get('api_key', '')),
            ('validate_checkbox', download.get('validate_download', True)),
            ('backup_checkbox', download.get('backup_existing', False))
        ]
        
        [self._set_widget_value(ui_components, widget_name, value) for widget_name, value in widget_updates]
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config dengan comprehensive checks"""
        errors = []
        
        # Extract roboflow config
        roboflow = config.get('data', {}).get('roboflow', {})
        
        # Required fields validation
        required_fields = {
            'workspace': roboflow.get('workspace', '').strip(),
            'project': roboflow.get('project', '').strip(),
            'version': roboflow.get('version', '').strip(),
            'api_key': roboflow.get('api_key', '').strip()
        }
        
        # Check missing required fields
        missing_fields = [field for field, value in required_fields.items() if not value]
        if missing_fields:
            errors.extend([f"Field '{field}' wajib diisi" for field in missing_fields])
        
        # Format validation
        if required_fields['workspace'] and len(required_fields['workspace']) < 3:
            errors.append("Workspace minimal 3 karakter")
        
        if required_fields['project'] and len(required_fields['project']) < 3:
            errors.append("Project minimal 3 karakter")
        
        if required_fields['api_key'] and len(required_fields['api_key']) < 10:
            errors.append("API key terlalu pendek")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': []
        }
    
    def _get_widget_value(self, ui_components: Dict[str, Any], widget_name: str, default_value: Any = None) -> Any:
        """Get widget value dengan safe access dan default"""
        widget = ui_components.get(widget_name)
        return getattr(widget, 'value', default_value) if widget else default_value
    
    def _set_widget_value(self, ui_components: Dict[str, Any], widget_name: str, value: Any) -> None:
        """Set widget value dengan safe error handling"""
        widget = ui_components.get(widget_name)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = value
            except Exception:
                pass  # Silent fail untuk widget update issues

# Factory function
def create_downloader_config_handler() -> DownloaderConfigHandler:
    """Factory untuk membuat downloader config handler"""
    return DownloaderConfigHandler()