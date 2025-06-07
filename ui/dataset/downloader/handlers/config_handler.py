"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: FIXED config handler dengan proper file save ke dataset_config.yaml
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config
from smartcash.common.config.manager import get_config_manager

class DownloaderConfigHandler(ConfigHandler):
    """FIXED config handler dengan proper file save functionality"""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'dataset_config.yaml'  # Explicitly use dataset_config.yaml
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components"""
        return self.extract_config_from_ui(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        self.update_ui_from_config(ui_components, config)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan API key auto-detection"""
        config = get_default_downloader_config()
        return set_api_key_to_config(config, force_refresh=False)
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """FIXED save config ke dataset_config.yaml dengan proper merging"""
        try:
            # Use dataset_config.yaml sebagai target file
            filename = config_filename or self.config_filename
            
            # Extract current config dari UI
            current_config = self.extract_config_from_ui(ui_components)
            
            # Load existing config dari file untuk merge
            existing_config = self.config_manager.load_config(filename)
            
            # FIXED: Merge config dengan strategy yang benar
            merged_config = self._merge_downloader_config(existing_config, current_config)
            
            # Save merged config ke file
            success = self.config_manager.save_config(merged_config, filename)
            
            if success:
                self.logger.success(f"✅ Config tersimpan ke {filename}")
                return True
            else:
                self.logger.error(f"❌ Gagal menyimpan config ke {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {str(e)}")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """FIXED reset config dengan proper default loading"""
        try:
            # Get default config
            default_config = self.get_default_config()
            
            # Update UI dengan default
            self.update_ui_from_config(ui_components, default_config)
            
            # Save default ke file
            filename = config_filename or self.config_filename
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self.logger.success(f"🔄 Config direset ke default dan tersimpan ke {filename}")
                return True
            else:
                self.logger.warning("⚠️ Config direset di UI tapi gagal tersimpan ke file")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error reset config: {str(e)}")
            return False
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """FIXED load config dari dataset_config.yaml dengan fallback"""
        try:
            filename = config_filename or self.config_filename
            
            # Load dari file
            config = self.config_manager.load_config(filename)
            
            if not config:
                self.logger.info(f"📂 File {filename} tidak ditemukan, menggunakan default")
                config = self.get_default_config()
                
                # Save default ke file untuk pertama kali
                self.config_manager.save_config(config, filename)
                self.logger.info(f"💾 Default config tersimpan ke {filename}")
            
            # Auto-detect API key jika kosong
            config = set_api_key_to_config(config, force_refresh=False)
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def _merge_downloader_config(self, existing: Dict[str, Any], new_downloader: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED merge downloader config dengan existing dataset_config.yaml"""
        # Start dengan existing config
        merged = dict(existing) if existing else {}
        
        # Extract downloader-specific sections dari new config
        downloader_sections = ['data', 'download', 'uuid_renaming']
        
        for section in downloader_sections:
            if section in new_downloader:
                if section == 'data':
                    # Merge data section dengan hati-hati
                    merged.setdefault('data', {})
                    
                    # Merge roboflow config
                    if 'roboflow' in new_downloader['data']:
                        merged['data']['roboflow'] = new_downloader['data']['roboflow']
                    
                    # Merge file_naming config
                    if 'file_naming' in new_downloader['data']:
                        merged['data']['file_naming'] = new_downloader['data']['file_naming']
                    
                    # Keep existing data source
                    merged['data']['source'] = new_downloader['data'].get('source', 'roboflow')
                    
                else:
                    # Replace section completely untuk download dan uuid_renaming
                    merged[section] = new_downloader[section]
        
        return merged
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components dengan proper structure"""
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