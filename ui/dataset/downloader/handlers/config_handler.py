"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Deskripsi: Fixed config handler dengan proper reset functionality dan error handling
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.downloader.handlers.config_extractor import extract_downloader_config
from smartcash.ui.dataset.downloader.handlers.config_updater import update_downloader_ui, validate_ui_inputs
from smartcash.ui.dataset.downloader.utils.colab_secrets import set_api_key_to_config, get_api_key_from_secrets
from smartcash.common.config.manager import get_config_manager

class DownloaderConfigHandler(ConfigHandler):
    """Fixed config handler dengan proper reset dan API key handling"""
    
    def __init__(self, module_name: str = 'downloader', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'dataset_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari downloader UI components"""
        return extract_downloader_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_downloader_ui(ui_components, config)
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan API key auto-detection dan fallback ke defaults"""
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
            
            # Auto-detect dan set API key dari Colab secrets
            config = set_api_key_to_config(config, force_refresh=False)
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan merge strategy untuk dataset_config.yaml"""
        try:
            filename = config_filename or self.config_filename
            
            # Extract current config dari UI
            current_config = self.extract_config(ui_components)
            
            # Validate sebelum save
            validation = self.validate_config(current_config)
            if not validation['valid']:
                self.logger.error(f"❌ Config tidak valid: {'; '.join(validation['errors'])}")
                return False
            
            # Load existing config untuk merge
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Merge dengan strategy yang aman
            merged_config = self._merge_downloader_config(existing_config, current_config)
            
            # Save merged config
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
        """Enhanced reset config dengan proper UI update dan error handling"""
        try:
            self.logger.info("🔄 Starting config reset...")
            
            # Preserve current API key dari UI
            current_api_key = ''
            api_key_widget = ui_components.get('api_key_input')
            if api_key_widget and hasattr(api_key_widget, 'value'):
                current_api_key = api_key_widget.value.strip()
                self.logger.info("🔒 Preserving current API key from UI")
            
            # Auto-detect dari Colab secrets jika UI kosong
            if not current_api_key:
                detected_key = get_api_key_from_secrets()
                if detected_key:
                    current_api_key = detected_key
                    self.logger.info("🔑 API key auto-detected dari Colab secrets")
            
            # Get default config dengan proper error handling
            try:
                default_config = self.get_default_config()
                self.logger.info("✅ Default config loaded")
            except Exception as e:
                self.logger.error(f"❌ Error getting default config: {str(e)}")
                return False
            
            # Preserve API key di config
            if current_api_key:
                if 'data' not in default_config:
                    default_config['data'] = {}
                if 'roboflow' not in default_config['data']:
                    default_config['data']['roboflow'] = {}
                default_config['data']['roboflow']['api_key'] = current_api_key
                self.logger.info("🔒 API key preserved in default config")
            else:
                # Auto-detect dari Colab secrets
                try:
                    default_config = set_api_key_to_config(default_config, force_refresh=False)
                    self.logger.info("🔍 Attempted auto-detection from Colab secrets")
                except Exception as e:
                    self.logger.warning(f"⚠️ Auto-detection failed: {str(e)}")
            
            # Update UI dengan default config yang sudah dimodifikasi
            try:
                self.logger.info("🔄 Updating UI components with default config...")
                self.update_ui(ui_components, default_config)
                self.logger.success("✅ UI components updated successfully")
            except Exception as e:
                self.logger.error(f"❌ Error updating UI: {str(e)}")
                return False
            
            # Save default ke file
            try:
                filename = config_filename or self.config_filename
                success = self.config_manager.save_config(default_config, filename)
                
                if success:
                    api_status = "dengan API key" if current_api_key else "tanpa API key"
                    self.logger.success(f"🔄 Config berhasil direset ke default {api_status}")
                    return True
                else:
                    self.logger.warning("⚠️ Config direset di UI tapi gagal tersimpan ke file")
                    return True  # UI sudah direset, anggap berhasil
            except Exception as e:
                self.logger.error(f"❌ Error saving config: {str(e)}")
                return True  # UI sudah direset, anggap berhasil
                
        except Exception as e:
            self.logger.error(f"❌ Error reset config: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation dengan comprehensive checks untuk downloader"""
        errors = []
        warnings = []
        
        # Extract roboflow config
        roboflow = config.get('data', {}).get('roboflow', {})
        download = config.get('download', {})
        
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
            errors.append("API key terlalu pendek (minimal 10 karakter)")
        
        # Download config validation
        retry_count = download.get('retry_count', 3)
        if not isinstance(retry_count, int) or retry_count < 1 or retry_count > 10:
            warnings.append("Retry count sebaiknya antara 1-10")
        
        timeout = download.get('timeout', 30)
        if not isinstance(timeout, int) or timeout < 10 or timeout > 300:
            warnings.append("Timeout sebaiknya antara 10-300 detik")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _merge_downloader_config(self, existing: Dict[str, Any], new_downloader: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced merge downloader config dengan existing dataset_config.yaml"""
        # Start dengan existing config atau empty dict
        merged = dict(existing) if existing else {}
        
        # Downloader-specific sections yang akan di-merge
        downloader_sections = ['data', 'download', 'uuid_renaming', 'validation', 'cleanup']
        
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
                    
                    # Merge local paths jika ada
                    if 'local' in new_downloader['data']:
                        merged['data']['local'] = new_downloader['data']['local']
                    
                    # Keep existing data source dan dir
                    merged['data']['source'] = new_downloader['data'].get('source', 'roboflow')
                    merged['data']['dir'] = new_downloader['data'].get('dir', 'data')
                    
                else:
                    # Replace section completely untuk download, uuid_renaming, validation, cleanup
                    merged[section] = new_downloader[section]
        
        # Preserve config metadata
        merged['config_version'] = new_downloader.get('config_version', '1.0')
        merged['updated_at'] = new_downloader.get('updated_at')
        merged['_base_'] = new_downloader.get('_base_', 'base_config.yaml')
        
        return merged
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper untuk extract config dari UI dengan validation"""
        try:
            return self.extract_config(ui_components)
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {str(e)}")
            return self.get_default_config()
    
    def get_api_key_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Get API key status dengan auto-detection info"""
        try:
            from smartcash.ui.dataset.downloader.utils.colab_secrets import validate_api_key
            
            # Check current UI value
            current_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
            
            # Check Colab secrets
            detected_key = get_api_key_from_secrets()
            
            if detected_key:
                validation = validate_api_key(detected_key)
                return {
                    'source': 'colab_secret',
                    'valid': validation['valid'],
                    'message': f"Auto-detect dari Colab: {validation['message']}",
                    'key_preview': f"{detected_key[:4]}...{detected_key[-4:]}" if len(detected_key) > 8 else '****'
                }
            elif current_key:
                validation = validate_api_key(current_key)
                return {
                    'source': 'manual_input',
                    'valid': validation['valid'],
                    'message': f"Manual input: {validation['message']}",
                    'key_preview': f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else '****'
                }
            else:
                return {
                    'source': 'not_provided',
                    'valid': False,
                    'message': 'API key belum diisi',
                    'key_preview': '****'
                }
                
        except Exception as e:
            return {
                'source': 'error',
                'valid': False,
                'message': f"Error checking API key: {str(e)}",
                'key_preview': '****'
            }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan optimal workers dan proper structure"""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
            from smartcash.common.threadpools import get_download_workers, get_rename_workers, optimal_io_workers, get_optimal_thread_count
            
            default_config = get_default_downloader_config()
            
            # Update dengan optimal workers
            default_config['download']['max_workers'] = get_download_workers()
            default_config['uuid_renaming']['parallel_workers'] = get_rename_workers(5000)
            
            # Add missing sections if not present
            if 'validation' not in default_config:
                default_config['validation'] = {
                    'enabled': True,
                    'parallel_workers': get_optimal_thread_count('io')
                }
            
            if 'cleanup' not in default_config:
                default_config['cleanup'] = {
                    'auto_cleanup_downloads': False,
                    'parallel_workers': optimal_io_workers()
                }
            
            return default_config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting default config: {str(e)}")
            # Fallback minimal config
            return {
                'data': {
                    'source': 'roboflow',
                    'roboflow': {
                        'workspace': 'smartcash-wo2us',
                        'project': 'rupiah-emisi-2022',
                        'version': '3',
                        'api_key': '',
                        'output_format': 'yolov5pytorch'
                    }
                },
                'download': {
                    'rename_files': True,
                    'validate_download': True,
                    'backup_existing': False
                }
            }