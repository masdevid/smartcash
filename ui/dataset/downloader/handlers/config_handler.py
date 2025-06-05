"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler dengan proper widget access dan API key auto-detection
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, validate_api_key, get_api_key_from_secrets

class DownloadConfigHandler(ConfigHandler):
    """Fixed config handler dengan proper widget access dan auto API key detection"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI widgets dengan proper access pattern"""
        config = {
            'workspace': getattr(ui_components.get('workspace_input'), 'value', '').strip(),
            'project': getattr(ui_components.get('project_input'), 'value', '').strip(), 
            'version': getattr(ui_components.get('version_input'), 'value', '').strip(),
            'api_key': getattr(ui_components.get('api_key_input'), 'value', '').strip(),
            'output_format': 'yolov5pytorch',
            'validate_download': getattr(ui_components.get('validate_checkbox'), 'value', True),
            'organize_dataset': True,  # Always true, no checkbox
            'backup_existing': getattr(ui_components.get('backup_checkbox'), 'value', False)
        }
        
        # Auto-detect API key jika kosong
        if not config['api_key']:
            detected_key = get_api_key_from_secrets()
            if detected_key:
                config['api_key'] = detected_key
                config['_api_key_source'] = 'colab_secret'
                # Update UI widget juga
                if 'api_key_input' in ui_components:
                    ui_components['api_key_input'].value = detected_key
        
        # Validate API key
        if config['api_key']:
            validation = validate_api_key(config['api_key'])
            config['_api_key_valid'] = validation['valid']
            config['_api_key_message'] = validation['message']
            config['_api_key_source'] = config.get('_api_key_source', 'manual')
        
        self._current_config = config
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan auto API key refresh"""
        # Auto-detect API key untuk reset
        if not config.get('api_key'):
            detected_key = get_api_key_from_secrets()
            if detected_key:
                config['api_key'] = detected_key
                config['_api_key_source'] = 'colab_secret'
        
        # Update widgets dengan proper attribute access
        widget_updates = [
            ('workspace_input', config.get('workspace', 'smartcash-wo2us')),
            ('project_input', config.get('project', 'rupiah-emisi-2022')),
            ('version_input', config.get('version', '3')),
            ('api_key_input', config.get('api_key', '')),
            ('validate_checkbox', config.get('validate_download', True)),
            ('backup_checkbox', config.get('backup_existing', False))
        ]
        
        [setattr(ui_components[widget_key], 'value', value) 
         for widget_key, value in widget_updates 
         if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]
        
        self._current_config = config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API key auto-detection"""
        config = get_default_download_config()
        
        # Auto-detect API key
        detected_key = get_api_key_from_secrets()
        if detected_key:
            config['api_key'] = detected_key
            config['_api_key_source'] = 'colab_secret'
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config dengan comprehensive checks"""
        errors = []
        
        # Required field validation dengan one-liner
        required_checks = [
            (config.get('workspace', '').strip(), 'Workspace wajib diisi'),
            (config.get('project', '').strip(), 'Project wajib diisi'),
            (config.get('version', '').strip(), 'Version wajib diisi'),
            (config.get('api_key', '').strip(), 'API key wajib diisi')
        ]
        
        [errors.append(message) for value, message in required_checks if not value]
        
        # API key validation
        api_key = config.get('api_key', '').strip()
        if api_key:
            validation = validate_api_key(api_key)
            not validation['valid'] and errors.append(f"API key tidak valid: {validation['message']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    # Enhanced hooks dengan UI integration
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah save berhasil dengan UI notification"""
        super().after_save_success(ui_components, config)
        
        dataset_id = f"{config.get('workspace', '')}/{config.get('project', '')}:v{config.get('version', '')}"
        api_source = config.get('_api_key_source', 'manual')
        
        # Show success di status panel dan log
        success_msg = f"✅ Konfigurasi berhasil disimpan: {dataset_id} (API: {api_source})"
        self._show_notification(ui_components, success_msg, 'success')
        self.logger.success(f"💾 {success_msg}")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah save gagal dengan UI notification"""
        super().after_save_failure(ui_components, error)
        
        error_msg = f"❌ Gagal menyimpan konfigurasi: {error}"
        self._show_notification(ui_components, error_msg, 'error')
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah reset berhasil dengan UI notification"""
        super().after_reset_success(ui_components, config)
        
        api_source = config.get('_api_key_source', 'default')
        success_msg = f"✅ Konfigurasi berhasil direset ke default (API: {api_source})"
        self._show_notification(ui_components, success_msg, 'success')
        self.logger.success(f"🔄 {success_msg}")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah reset gagal dengan UI notification"""
        super().after_reset_failure(ui_components, error)
        
        error_msg = f"❌ Gagal reset konfigurasi: {error}"
        self._show_notification(ui_components, error_msg, 'error')
    
    def _show_notification(self, ui_components: Dict[str, Any], message: str, msg_type: str) -> None:
        """Show notification di status panel dan log output dengan one-liner"""
        # Update status panel
        status_colors = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}
        color = status_colors.get(msg_type, '#007bff')
        
        status_html = f"""<div style="padding: 12px; background-color: {color}15; border-left: 4px solid {color}; border-radius: 4px; margin-bottom: 15px;"><span style="color: {color}; font-weight: 500;">{message}</span></div>"""
        
        status_panel = ui_components.get('status_panel')
        status_panel and setattr(status_panel, 'value', status_html)
        
        # Show di log output
        log_output = ui_components.get('log_output')
        if log_output:
            with log_output:
                from IPython.display import display, HTML
                display(HTML(f"""<div style="color: {color}; margin: 5px 0; padding: 8px; background: {color}10; border-radius: 4px;">{message}</div>"""))

# Factory function
def create_download_config_handler(parent_module: str = 'dataset') -> DownloadConfigHandler:
    """Factory untuk create download config handler"""
    return DownloadConfigHandler('downloader', parent_module)