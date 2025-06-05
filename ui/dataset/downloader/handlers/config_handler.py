"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Updated config handler tanpa format_dropdown (hardcoded yolov5pytorch)
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, validate_api_key

class DownloadConfigHandler(ConfigHandler):
    """Config handler untuk download dengan API key management (format hardcoded)"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components dengan validation (format hardcoded ke yolov5pytorch)"""
        config = {
            'workspace': ui_components.get('workspace_input', {}).get('value', '').strip(),
            'project': ui_components.get('project_input', {}).get('value', '').strip(), 
            'version': ui_components.get('version_input', {}).get('value', '').strip(),
            'api_key': ui_components.get('api_key_input', {}).get('value', '').strip(),
            'output_format': 'yolov5pytorch',  # Hardcoded format
            'validate_download': ui_components.get('validate_checkbox', {}).get('value', True),
            'organize_dataset': ui_components.get('organize_checkbox', {}).get('value', True),
            'backup_existing': ui_components.get('backup_checkbox', {}).get('value', False)
        }
        
        # Validate dan set API key info
        if config['api_key']:
            validation = validate_api_key(config['api_key'])
            config['_api_key_valid'] = validation['valid']
            config['_api_key_message'] = validation['message']
            config['_api_key_source'] = 'manual'
        
        self._current_config = config
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan API key refresh (tanpa format_dropdown)"""
        # Refresh API key jika diperlukan
        config = set_api_key_to_config(config.copy())
        
        # Update UI widgets dengan one-liner (tanpa format_dropdown)
        widget_mappings = {
            'workspace_input': config.get('workspace', ''),
            'project_input': config.get('project', ''),
            'version_input': config.get('version', ''),
            'api_key_input': config.get('api_key', ''),
            'validate_checkbox': config.get('validate_download', True),
            'organize_checkbox': config.get('organize_dataset', True),
            'backup_checkbox': config.get('backup_existing', False)
        }
        
        [setattr(ui_components[widget_key], 'value', value) 
         for widget_key, value in widget_mappings.items() 
         if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]
        
        # Update API key info display jika ada
        if 'api_key_info' in ui_components:
            from smartcash.ui.dataset.downloader.utils.colab_secrets import create_api_key_info_html
            info_html = create_api_key_info_html(config)
            ui_components['api_key_info'].value = info_html
        
        self._current_config = config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API key injection (format hardcoded)"""
        config = get_default_download_config()
        config['output_format'] = 'yolov5pytorch'  # Ensure hardcoded format
        return set_api_key_to_config(config)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate download config dengan comprehensive checks (tanpa format validation)"""
        errors = []
        
        # Required fields validation
        required_fields = {
            'workspace': 'Workspace wajib diisi',
            'project': 'Project wajib diisi', 
            'version': 'Version wajib diisi',
            'api_key': 'API key wajib diisi'
        }
        
        [errors.append(message) for field, message in required_fields.items() 
         if not config.get(field, '').strip()]
        
        # API key validation
        api_key = config.get('api_key', '').strip()
        if api_key:
            validation = validate_api_key(api_key)
            if not validation['valid']:
                errors.append(f"API key tidak valid: {validation['message']}")
        
        # Format validation - ensure hardcoded format
        config['output_format'] = 'yolov5pytorch'
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': []
        }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config"""
        return self._current_config.copy()
    
    def refresh_api_key(self, ui_components: Dict[str, Any]) -> bool:
        """Refresh API key dari Colab Secret"""
        try:
            current_config = self.get_current_config()
            refreshed_config = set_api_key_to_config(current_config, force_refresh=True)
            
            if refreshed_config.get('api_key') != current_config.get('api_key'):
                self.update_ui(ui_components, refreshed_config)
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error refresh API key: {str(e)}")
        
        return False
    
    def get_dataset_identifier(self) -> str:
        """Get dataset identifier untuk logging"""
        config = self.get_current_config()
        workspace = config.get('workspace', '').strip()
        project = config.get('project', '').strip() 
        version = config.get('version', '').strip()
        
        if not all([workspace, project, version]):
            return "dataset-belum-lengkap"
        
        return f"{workspace}/{project}:v{version}"
    
    def is_ready_for_download(self) -> Dict[str, Any]:
        """Check apakah config siap untuk download"""
        config = self.get_current_config()
        validation = self.validate_config(config)
        
        return {
            'ready': validation['valid'],
            'issues': validation['errors'],
            'dataset_id': self.get_dataset_identifier(),
            'has_api_key': bool(config.get('api_key', '').strip()),
            'api_key_source': config.get('_api_key_source', 'unknown'),
            'format': 'yolov5pytorch'  # Always hardcoded
        }
    
    # Hook overrides dengan enhanced functionality
    def before_save(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum save dengan API key validation"""
        super().before_save(ui_components)
        
        # Extract dan validate config sebelum save
        config = self.extract_config(ui_components)
        validation = self.validate_config(config)
        
        if not validation['valid']:
            self._update_status_panel(ui_components, 
                                    f"⚠️ Config tidak valid: {'; '.join(validation['errors'])}", 
                                    "warning")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah save berhasil dengan dataset info"""
        super().after_save_success(ui_components, config)
        
        dataset_id = self.get_dataset_identifier()
        api_source = config.get('_api_key_source', 'unknown')
        
        self.logger.success(f"💾 Download config saved: {dataset_id} (API: {api_source}, Format: YOLOv5)")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah reset berhasil dengan API key refresh"""
        super().after_reset_success(ui_components, config)
        
        # Auto refresh API key setelah reset
        if config.get('_api_key_source') == 'colab_secret':
            self.logger.info("🔄 API key di-refresh dari Colab Secret")

# Factory function
def create_download_config_handler(parent_module: str = 'dataset') -> DownloadConfigHandler:
    """Factory untuk create download config handler"""
    return DownloadConfigHandler('downloader', parent_module)