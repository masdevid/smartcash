"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler dengan variable scoping fixes untuk log_output errors
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, validate_api_key, get_api_key_from_secrets

class DownloadConfigHandler(ConfigHandler):
    """Fixed config handler dengan proper variable scoping"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI widgets dengan safe value access"""
        config = {
            'workspace': self._get_widget_value(ui_components, 'workspace_input', '').strip(),
            'project': self._get_widget_value(ui_components, 'project_input', '').strip(), 
            'version': self._get_widget_value(ui_components, 'version_input', '').strip(),
            'api_key': self._get_widget_value(ui_components, 'api_key_input', '').strip(),
            'output_format': 'yolov5pytorch',  # Hardcoded
            'validate_download': self._get_widget_value(ui_components, 'validate_checkbox', True),
            'organize_dataset': True,  # Always true
            'backup_existing': self._get_widget_value(ui_components, 'backup_checkbox', False),
            
            # Roboflow section untuk persistence
            'roboflow': {
                'workspace': self._get_widget_value(ui_components, 'workspace_input', '').strip(),
                'project': self._get_widget_value(ui_components, 'project_input', '').strip(),
                'version': self._get_widget_value(ui_components, 'version_input', '').strip(),
                'api_key': self._get_widget_value(ui_components, 'api_key_input', '').strip(),
            }
        }
        
        # Auto-detect API key jika kosong
        if not config['api_key']:
            detected_key = get_api_key_from_secrets()
            if detected_key:
                config['api_key'] = detected_key
                config['_api_key_source'] = 'colab_secret'
                config['roboflow']['api_key'] = detected_key
                # Update UI widget jika ada
                api_key_widget = ui_components.get('api_key_input')
                if api_key_widget and hasattr(api_key_widget, 'value'):
                    api_key_widget.value = detected_key
        
        # Validate API key
        if config['api_key']:
            validation = validate_api_key(config['api_key'])
            config['_api_key_valid'] = validation['valid']
            config['_api_key_message'] = validation['message']
            config['_api_key_source'] = config.get('_api_key_source', 'manual')
        
        self._current_config = config
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan safe widget access"""
        # Auto-detect API key untuk reset
        if not config.get('api_key'):
            detected_key = get_api_key_from_secrets()
            if detected_key:
                config['api_key'] = detected_key
                config['_api_key_source'] = 'colab_secret'
        
        # Extract roboflow config
        roboflow_config = config.get('roboflow', {})
        
        # Widget updates dengan safe access
        widget_updates = [
            ('workspace_input', roboflow_config.get('workspace', config.get('workspace', 'smartcash-wo2us'))),
            ('project_input', roboflow_config.get('project', config.get('project', 'rupiah-emisi-2022'))),
            ('version_input', roboflow_config.get('version', config.get('version', '3'))),
            ('api_key_input', roboflow_config.get('api_key', config.get('api_key', ''))),
            ('validate_checkbox', config.get('validate_download', True)),
            ('backup_checkbox', config.get('backup_existing', False))
        ]
        
        # Update widget values dengan safe access
        for widget_key, value in widget_updates:
            self._set_widget_value(ui_components, widget_key, value)
        
        self._current_config = config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan auto-detect API key"""
        config = get_default_download_config()
        
        # Auto-detect API key
        detected_key = get_api_key_from_secrets()
        if detected_key:
            config['api_key'] = detected_key
            config['_api_key_source'] = 'colab_secret'
            roboflow_config = config.get('roboflow', {})
            roboflow_config['api_key'] = detected_key
            config['roboflow'] = roboflow_config
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config dengan comprehensive checks"""
        errors = []
        
        # Required field validation
        required_checks = [
            (config.get('workspace', '').strip(), 'Workspace wajib diisi'),
            (config.get('project', '').strip(), 'Project wajib diisi'),
            (config.get('version', '').strip(), 'Version wajib diisi'),
            (config.get('api_key', '').strip(), 'API key wajib diisi')
        ]
        
        for value, message in required_checks:
            if not value:
                errors.append(message)
        
        # API key validation
        api_key = config.get('api_key', '').strip()
        if api_key:
            validation = validate_api_key(api_key)
            if not validation['valid']:
                errors.append(f"API key tidak valid: {validation['message']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def save_config(self, ui_components: Dict[str, Any], config_name: str = None) -> bool:
        """Save config dengan fixed variable scoping"""
        try:
            self.before_save(ui_components)
            config = self.extract_config(ui_components)
            
            # Streamlined config structure
            persistent_config = {
                **config,
                'module_name': self.module_name,
                'version': '1.0.0',
                'created_by': 'SmartCash Download Module',
                'last_updated': __import__('datetime').datetime.now().isoformat(),
                
                # Roboflow section persistence
                'roboflow': {
                    'workspace': config.get('workspace', ''),
                    'project': config.get('project', ''),
                    'version': config.get('version', ''),
                    'api_key': config.get('api_key', '')
                }
            }
            
            success = self.config_manager.save_config(persistent_config, config_name or f"{self.module_name}_config")
            
            if success:
                ui_components['config'] = persistent_config
                self.after_save_success(ui_components, persistent_config)
                # Trigger callbacks dengan safe execution
                for callback in self.callbacks:
                    try:
                        callback(persistent_config)
                    except Exception as e:
                        self.logger.debug(f"🔍 Callback error: {str(e)}")
            else:
                self.after_save_failure(ui_components, "Gagal menyimpan konfigurasi")
            
            return success
            
        except Exception as e:
            self.after_save_failure(ui_components, str(e))
            return False
    
    def load_config(self, config_name: str = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan enhanced processing"""
        config_name = config_name or f"{self.module_name}_config"
        
        try:
            # Load specific config
            specific_config = self.config_manager.get_config(config_name)
            
            if specific_config:
                enhanced_config = self._enhance_loaded_config(specific_config)
                self.logger.info(f"📄 Loaded config: {config_name}")
                return enhanced_config
            
            # Fallback ke base_config.yaml
            if use_base_config:
                base_config = self.config_manager.get_config('base_config')
                if base_config:
                    enhanced_base = self._enhance_loaded_config(base_config)
                    self.logger.info(f"📄 Loaded base_config.yaml for {config_name}")
                    return enhanced_base
            
            # Final fallback
            default_config = self.get_default_config()
            self.logger.warning(f"⚠️ Using defaults for {config_name}")
            return default_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def _enhance_loaded_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance loaded config dengan auto-detect API key"""
        roboflow_section = config.get('roboflow', {})
        
        # Enhanced config structure
        enhanced = {
            **config,
            'workspace': roboflow_section.get('workspace', config.get('workspace', 'smartcash-wo2us')),
            'project': roboflow_section.get('project', config.get('project', 'rupiah-emisi-2022')),
            'version': roboflow_section.get('version', config.get('version', '3')),
            'api_key': roboflow_section.get('api_key', config.get('api_key', ''))
        }
        
        # Auto-detect API key jika tidak ada
        if not enhanced['api_key']:
            detected_key = get_api_key_from_secrets()
            if detected_key:
                enhanced['api_key'] = detected_key
                enhanced['_api_key_source'] = 'colab_secret'
        
        return enhanced
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Fixed save success hook dengan safe log output access"""
        super().after_save_success(ui_components, config)
        
        dataset_id = f"{config.get('workspace', '')}/{config.get('project', '')}:v{config.get('version', '')}"
        api_source = config.get('_api_key_source', 'manual')
        
        success_msg = f"✅ Konfigurasi tersimpan: {dataset_id} (API: {api_source})"
        self._show_notification_safe(ui_components, success_msg, 'success')
        self.logger.success(f"💾 {success_msg}")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Fixed reset success hook dengan safe log output access"""
        super().after_reset_success(ui_components, config)
        
        api_source = config.get('_api_key_source', 'default')
        success_msg = f"✅ Konfigurasi direset ke default (API: {api_source})"
        self._show_notification_safe(ui_components, success_msg, 'success')
        self.logger.success(f"🔄 {success_msg}")
    
    def _show_notification_safe(self, ui_components: Dict[str, Any], message: str, msg_type: str) -> None:
        """Show notification dengan safe log output access"""
        status_colors = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}
        color = status_colors.get(msg_type, '#007bff')
        
        status_html = f"""<div style="padding: 12px; background-color: {color}15; border-left: 4px solid {color}; border-radius: 4px; margin-bottom: 15px;"><span style="color: {color}; font-weight: 500;">{message}</span></div>"""
        
        # Safe status panel update
        status_panel = ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            status_panel.value = status_html
        
        # Safe log output display
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            try:
                with log_output:
                    from IPython.display import display, HTML
                    display(HTML(f"""<div style="color: {color}; margin: 5px 0; padding: 8px; background: {color}10; border-radius: 4px;">{message}</div>"""))
            except Exception as e:
                self.logger.debug(f"🔍 Log output display error: {str(e)}")
    
    def _get_widget_value(self, ui_components: Dict[str, Any], widget_key: str, default_value=None):
        """Safe widget value getter"""
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            return widget.value
        return default_value
    
    def _set_widget_value(self, ui_components: Dict[str, Any], widget_key: str, value):
        """Safe widget value setter"""
        widget = ui_components.get(widget_key)
        if widget and hasattr(widget, 'value'):
            try:
                widget.value = value
            except Exception as e:
                self.logger.debug(f"🔍 Widget update error for {widget_key}: {str(e)}")

# Factory function
def create_download_config_handler(parent_module: str = 'dataset') -> DownloadConfigHandler:
    """Factory untuk fixed download config handler"""
    return DownloadConfigHandler('downloader', parent_module)