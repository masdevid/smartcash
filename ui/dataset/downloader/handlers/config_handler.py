"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Enhanced config handler dengan path fields dan fixed persistence untuk backup/preprocessed dirs
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, validate_api_key, get_api_key_from_secrets

class DownloadConfigHandler(ConfigHandler):
    """Enhanced config handler dengan path fields dan fixed persistence untuk backup/preprocessed dirs"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._current_config = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI widgets dengan enhanced path fields dan one-liner attribute access"""
        config = {
            'workspace': getattr(ui_components.get('workspace_input'), 'value', '').strip(),
            'project': getattr(ui_components.get('project_input'), 'value', '').strip(), 
            'version': getattr(ui_components.get('version_input'), 'value', '').strip(),
            'api_key': getattr(ui_components.get('api_key_input'), 'value', '').strip(),
            'output_format': 'yolov5pytorch',  # Hardcoded
            'validate_download': getattr(ui_components.get('validate_checkbox'), 'value', True),
            'organize_dataset': True,  # Always true, no checkbox
            'backup_existing': getattr(ui_components.get('backup_checkbox'), 'value', False),
            
            # Enhanced path configuration dengan one-liner fallback
            'backup_dir': getattr(ui_components.get('backup_dir_input'), 'value', 'data/backup').strip(),
            'preprocessed_dir': getattr(ui_components.get('preprocessed_dir_input'), 'value', 'data/preprocessed').strip(),
            
            # Roboflow section untuk persistence dengan one-liner nested structure
            'roboflow': {
                'workspace': getattr(ui_components.get('workspace_input'), 'value', '').strip(),
                'project': getattr(ui_components.get('project_input'), 'value', '').strip(),
                'version': getattr(ui_components.get('version_input'), 'value', '').strip(),
                'api_key': getattr(ui_components.get('api_key_input'), 'value', '').strip(),
            },
            
            # Paths section untuk persistence
            'paths': {
                'backup': getattr(ui_components.get('backup_dir_input'), 'value', 'data/backup').strip(),
                'preprocessed': getattr(ui_components.get('preprocessed_dir_input'), 'value', 'data/preprocessed').strip(),
            }
        }
        
        # Auto-detect API key jika kosong dengan one-liner detection dan update
        (not config['api_key'] and (detected_key := get_api_key_from_secrets()) and 
         config.update({'api_key': detected_key, '_api_key_source': 'colab_secret'}) and
         config['roboflow'].update({'api_key': detected_key}) and
         ui_components.get('api_key_input') and setattr(ui_components['api_key_input'], 'value', detected_key))
        
        # Validate API key dengan one-liner conditional validation
        config['api_key'] and config.update({
            '_api_key_valid': (validation := validate_api_key(config['api_key']))['valid'],
            '_api_key_message': validation['message'],
            '_api_key_source': config.get('_api_key_source', 'manual')
        })
        
        self._current_config = config
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan enhanced path fields dan auto API key refresh"""
        # Auto-detect API key untuk reset dengan one-liner conditional
        (not config.get('api_key') and (detected_key := get_api_key_from_secrets()) and 
         config.update({'api_key': detected_key, '_api_key_source': 'colab_secret'}))
        
        # Extract roboflow dan paths config dengan one-liner fallback
        roboflow_config = config.get('roboflow', {})
        paths_config = config.get('paths', {})
        
        # Enhanced widget updates dengan path fields - one-liner setattr calls
        widget_updates = [
            ('workspace_input', roboflow_config.get('workspace', config.get('workspace', 'smartcash-wo2us'))),
            ('project_input', roboflow_config.get('project', config.get('project', 'rupiah-emisi-2022'))),
            ('version_input', roboflow_config.get('version', config.get('version', '3'))),
            ('api_key_input', roboflow_config.get('api_key', config.get('api_key', ''))),
            ('validate_checkbox', config.get('validate_download', True)),
            ('backup_checkbox', config.get('backup_existing', False)),
            ('backup_dir_input', paths_config.get('backup', config.get('backup_dir', 'data/backup'))),
            ('preprocessed_dir_input', paths_config.get('preprocessed', config.get('preprocessed_dir', 'data/preprocessed')))
        ]
        
        # One-liner widget value updates dengan safety checks
        [setattr(ui_components[widget_key], 'value', value) 
         for widget_key, value in widget_updates 
         if widget_key in ui_components and hasattr(ui_components[widget_key], 'value')]
        
        self._current_config = config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API key auto-detection dan enhanced paths"""
        config = get_default_download_config()
        
        # Auto-detect API key dengan one-liner conditional
        (detected_key := get_api_key_from_secrets()) and config.update({
            'api_key': detected_key, 
            '_api_key_source': 'colab_secret',
            'roboflow': {**config.get('roboflow', {}), 'api_key': detected_key}
        })
        
        # Enhanced paths configuration dengan one-liner structure
        config.update({
            'backup_dir': 'data/backup',
            'preprocessed_dir': 'data/preprocessed',
            'paths': {
                'backup': 'data/backup',
                'preprocessed': 'data/preprocessed'
            }
        })
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config dengan enhanced path validation dan comprehensive checks"""
        errors = []
        
        # Required field validation dengan one-liner checks
        required_checks = [
            (config.get('workspace', '').strip(), 'Workspace wajib diisi'),
            (config.get('project', '').strip(), 'Project wajib diisi'),
            (config.get('version', '').strip(), 'Version wajib diisi'),
            (config.get('api_key', '').strip(), 'API key wajib diisi')
        ]
        
        [errors.append(message) for value, message in required_checks if not value]
        
        # API key validation dengan one-liner conditional
        (api_key := config.get('api_key', '').strip()) and not (validation := validate_api_key(api_key))['valid'] and errors.append(f"API key tidak valid: {validation['message']}")
        
        # Enhanced path validation dengan one-liner path checks
        path_checks = [
            (config.get('backup_dir', '').strip(), 'Backup directory wajib diisi'),
            (config.get('preprocessed_dir', '').strip(), 'Preprocessed directory wajib diisi')
        ]
        
        [errors.append(message) for value, message in path_checks if not value]
        
        # Path format validation dengan one-liner pattern matching
        import re
        path_pattern = r'^[a-zA-Z0-9_/.-]+$'
        path_validations = [
            (config.get('backup_dir', ''), 'backup_dir'),
            (config.get('preprocessed_dir', ''), 'preprocessed_dir')
        ]
        
        [errors.append(f"{field} format tidak valid - gunakan path relatif") 
         for path, field in path_validations 
         if path and not re.match(path_pattern, path)]
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def save_config(self, ui_components: Dict[str, Any], config_name: str = None) -> bool:
        """Override save dengan enhanced persistence untuk roboflow dan paths sections"""
        try:
            # Before save hook
            self.before_save(ui_components)
            
            # Extract config dengan enhanced fields
            config = self.extract_config(ui_components)
            
            # Enhanced config structure untuk proper persistence dengan one-liner nested updates
            persistent_config = {
                **config,
                'module_name': self.module_name,
                'version': '1.0.0',
                'created_by': 'SmartCash Download Module',
                'last_updated': __import__('datetime').datetime.now().isoformat(),
                
                # Ensure roboflow section persistence
                'roboflow': {
                    'workspace': config.get('workspace', ''),
                    'project': config.get('project', ''),
                    'version': config.get('version', ''),
                    'api_key': config.get('api_key', '')
                },
                
                # Ensure paths section persistence
                'paths': {
                    'backup': config.get('backup_dir', 'data/backup'),
                    'preprocessed': config.get('preprocessed_dir', 'data/preprocessed')
                }
            }
            
            # Save dengan enhanced config
            success = self.config_manager.save_config(persistent_config, config_name or f"{self.module_name}_config")
            
            if success:
                ui_components['config'] = persistent_config
                self.after_save_success(ui_components, persistent_config)
                [__import__('smartcash.ui.utils.fallback_utils', fromlist=['try_operation_safe']).try_operation_safe(lambda cb=cb: cb(persistent_config)) for cb in self.callbacks]
            else:
                self.after_save_failure(ui_components, "Gagal menyimpan konfigurasi")
            
            return success
            
        except Exception as e:
            self.after_save_failure(ui_components, str(e))
            return False
    
    def load_config(self, config_name: str = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Override load dengan enhanced persistence loading"""
        config_name = config_name or f"{self.module_name}_config"
        
        try:
            # Load specific config dengan enhanced fallback chain
            specific_config = self.config_manager.get_config(config_name)
            
            if specific_config:
                # Enhanced config processing dengan path validation
                enhanced_config = self._enhance_loaded_config(specific_config)
                self.logger.info(f"📄 Loaded enhanced config: {config_name}")
                return enhanced_config
            
            # Fallback ke base_config.yaml
            if use_base_config:
                base_config = self.config_manager.get_config('base_config')
                if base_config:
                    enhanced_base = self._enhance_loaded_config(base_config)
                    self.logger.info(f"📄 Loaded enhanced base_config.yaml for {config_name}")
                    return enhanced_base
            
            # Final fallback dengan enhanced defaults
            default_config = self.get_default_config()
            self.logger.warning(f"⚠️ Using enhanced defaults for {config_name}")
            return default_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading enhanced config: {str(e)}")
            return self.get_default_config()
    
    def _enhance_loaded_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance loaded config dengan path processing dan validation"""
        # Extract roboflow dan paths sections dengan one-liner fallback
        roboflow_section = config.get('roboflow', {})
        paths_section = config.get('paths', {})
        
        # Enhanced config dengan unified structure - one-liner merging
        enhanced = {
            **config,
            'workspace': roboflow_section.get('workspace', config.get('workspace', 'smartcash-wo2us')),
            'project': roboflow_section.get('project', config.get('project', 'rupiah-emisi-2022')),
            'version': roboflow_section.get('version', config.get('version', '3')),
            'api_key': roboflow_section.get('api_key', config.get('api_key', '')),
            'backup_dir': paths_section.get('backup', config.get('backup_dir', 'data/backup')),
            'preprocessed_dir': paths_section.get('preprocessed', config.get('preprocessed_dir', 'data/preprocessed'))
        }
        
        # Auto-detect API key jika tidak ada dengan one-liner conditional
        (not enhanced['api_key'] and (detected_key := get_api_key_from_secrets()) and 
         enhanced.update({'api_key': detected_key, '_api_key_source': 'colab_secret'}))
        
        return enhanced
    
    # Enhanced hooks dengan path configuration notification
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Enhanced hook setelah save berhasil dengan path info"""
        super().after_save_success(ui_components, config)
        
        dataset_id = f"{config.get('workspace', '')}/{config.get('project', '')}:v{config.get('version', '')}"
        api_source = config.get('_api_key_source', 'manual')
        backup_dir = config.get('backup_dir', 'data/backup')
        preprocessed_dir = config.get('preprocessed_dir', 'data/preprocessed')
        
        # Enhanced success message dengan path info
        success_msg = f"✅ Konfigurasi tersimpan: {dataset_id} (API: {api_source}) | Paths: backup={backup_dir}, preproc={preprocessed_dir}"
        self._show_notification(ui_components, success_msg, 'success')
        self.logger.success(f"💾 {success_msg}")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Enhanced hook setelah reset berhasil dengan path info"""
        super().after_reset_success(ui_components, config)
        
        api_source = config.get('_api_key_source', 'default')
        success_msg = f"✅ Konfigurasi direset ke default (API: {api_source}) dengan path configuration"
        self._show_notification(ui_components, success_msg, 'success')
        self.logger.success(f"🔄 {success_msg}")
    
    def _show_notification(self, ui_components: Dict[str, Any], message: str, msg_type: str) -> None:
        """Enhanced notification dengan one-liner color mapping dan display"""
        # One-liner status colors mapping
        status_colors = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}
        color = status_colors.get(msg_type, '#007bff')
        
        # One-liner HTML generation dan status panel update
        status_html = f"""<div style="padding: 12px; background-color: {color}15; border-left: 4px solid {color}; border-radius: 4px; margin-bottom: 15px;"><span style="color: {color}; font-weight: 500;">{message}</span></div>"""
        (status_panel := ui_components.get('status_panel')) and setattr(status_panel, 'value', status_html)
        
        # One-liner log output dengan HTML display
        (log_output := ui_components.get('log_output')) and (lambda: (
            __import__('IPython.display', fromlist=['display', 'HTML']),
            log_output.__enter__(),
            __import__('IPython.display', fromlist=['display']).display(__import__('IPython.display', fromlist=['HTML']).HTML(f"""<div style="color: {color}; margin: 5px 0; padding: 8px; background: {color}10; border-radius: 4px;">{message}</div>""")),
            log_output.__exit__(None, None, None)
        ))() if hasattr(log_output, '__enter__') else None

# Factory function
def create_enhanced_download_config_handler(parent_module: str = 'dataset') -> DownloadConfigHandler:
    """Factory untuk create enhanced download config handler dengan path support"""
    return DownloadConfigHandler('downloader', parent_module)