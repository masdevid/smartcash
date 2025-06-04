"""
File: smartcash/ui/dataset/download/handlers/download_config_handler.py
Deskripsi: ConfigHandler khusus untuk download module dengan environment-aware defaults
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
import os

class DownloadConfigHandler(ConfigHandler):
    """ConfigHandler khusus untuk download module dengan smart defaults."""
    
    def __init__(self, module_name: str = 'download', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.env_manager = get_environment_manager()
        self.paths = get_paths_for_environment(
            is_colab=self.env_manager.is_colab,
            is_drive_mounted=self.env_manager.is_drive_mounted
        )
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan smart field mapping."""
        field_mapping = {
            'workspace': 'workspace', 'project': 'project', 'version': 'version',
            'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
            'backup_checkbox': 'backup_before_download', 'organize_dataset': 'organize_dataset'
        }
        
        config = {}
        [config.update({config_key: getattr(ui_components.get(ui_key), 'value', None)})
         for ui_key, config_key in field_mapping.items()
         if ui_key in ui_components and hasattr(ui_components[ui_key], 'value')]
        
        # Don't save API key to config for security
        return {k: v for k, v in config.items() if v is not None}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan environment-aware values."""
        # Get smart defaults terlebih dahulu
        smart_config = self._merge_with_smart_defaults(config)
        
        field_mapping = {
            'workspace': 'workspace', 'project': 'project', 'version': 'version',
            'output_dir': 'output_dir', 'backup_dir': 'backup_dir',
            'backup_before_download': 'backup_checkbox', 'organize_dataset': 'organize_dataset'
        }
        
        [setattr(ui_components[ui_key], 'value', smart_config.get(config_key, ''))
         for config_key, ui_key in field_mapping.items()
         if ui_key in ui_components and hasattr(ui_components[ui_key], 'value')]
        
        # Handle API key separately (auto-detect)
        if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
            ui_components['api_key'].value = self._detect_api_key()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get environment-aware default config."""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'output_dir': self.paths['downloads'],
            'backup_dir': self.paths['backup'],
            'backup_before_download': False,
            'organize_dataset': True,
            'created_by': 'DownloadConfigHandler',
            'environment_type': 'Google Drive' if self.env_manager.is_drive_mounted else 'Local Storage'
        }
    
    def _merge_with_smart_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge config dengan smart defaults yang environment-aware."""
        defaults = self.get_default_config()
        merged = defaults.copy()
        merged.update(config)
        
        # Ensure paths are current
        merged['output_dir'] = merged.get('output_dir') or self.paths['downloads']
        merged['backup_dir'] = merged.get('backup_dir') or self.paths['backup']
        
        return merged
    
    def _detect_api_key(self) -> str:
        """Detect API key dari environment sources."""
        # Environment variable
        api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        if api_key: return api_key
        
        # Google Colab userdata
        try:
            from google.colab import userdata
            for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']:
                try:
                    api_key = userdata.get(key_name, '')
                    if api_key: return api_key
                except: continue
        except: pass
        
        return ''
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate download config dengan comprehensive checks."""
        errors = []
        
        # Required fields
        required = ['workspace', 'project', 'version']
        missing = [field for field in required if not config.get(field)]
        errors.extend([f"Field {field} wajib diisi" for field in missing])
        
        # Format validation
        if config.get('workspace') and len(config['workspace']) < 3:
            errors.append("Workspace ID terlalu pendek")
        
        if config.get('project') and len(config['project']) < 3:
            errors.append("Project ID terlalu pendek")
        
        # Path validation
        for path_key in ['output_dir', 'backup_dir']:
            if config.get(path_key):
                try:
                    from pathlib import Path
                    Path(config[path_key]).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Path {path_key} tidak valid: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get human-readable config summary."""
        workspace = config.get('workspace', 'N/A')
        project = config.get('project', 'N/A')
        version = config.get('version', 'N/A')
        storage = config.get('environment_type', 'Unknown')
        
        return f"📊 Download Config: {workspace}/{project}:{version} | Storage: {storage}"