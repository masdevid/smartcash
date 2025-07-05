"""
File: smartcash/ui/dataset/downloader/handlers/config_handler.py
Description: Config handler for downloader module with centralized error handling
"""

from typing import Dict, Any, Optional, Callable, TypeVar, cast, Union, List
import logging
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.dataset.downloader.handlers.base_downloader_handler import BaseDownloaderHandler
from smartcash.ui.dataset.downloader.handlers.config_extractor import extract_downloader_config
from smartcash.ui.dataset.downloader.handlers.config_updater import update_downloader_ui
from smartcash.ui.dataset.downloader.utils.validation_utils import validate_config
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.utils.colab_secrets import (
    set_api_key_to_config, 
    get_api_key_from_secrets,
    update_config_with_api_key,
    get_api_key_from_ui,
    detect_and_set_api_key
)
from smartcash.common.worker_utils import get_download_workers, get_rename_workers

class DownloaderConfigHandler(ConfigHandler, BaseDownloaderHandler):
    """Config handler for downloader module with centralized error handling and API key management.
    
    Features:
    - Centralized error handling with proper logging
    - API key auto-detection from Colab secrets
    - Support for non-persistent configuration
    - Proper merge strategy for dataset_config.yaml
    """
    
    @handle_ui_errors(error_component_title="Config Handler Initialization Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, module_name: str = 'downloader', 
                 parent_module: str = 'dataset', persistence_enabled: bool = True, 
                 use_shared_config: bool = True):
        """Initialize downloader config handler with centralized error handling.
        
        Args:
            ui_components: Dictionary containing UI components
            module_name: Name of the module
            parent_module: Parent module name
            persistence_enabled: Whether to enable config persistence to disk
            use_shared_config: Whether to use shared config manager
        """
        # Initialize both parent classes
        ConfigHandler.__init__(self, module_name=module_name, parent_module=parent_module,
                              persistence_enabled=persistence_enabled, use_shared_config=use_shared_config)
        BaseDownloaderHandler.__init__(self, ui_components=ui_components, module_name=module_name, 
                                      parent_module=parent_module)
        
        # Set config filename
        self.config_filename = 'dataset_config.yaml'
    
    @handle_ui_errors(error_component_title="Config Extraction Error", log_error=True, return_type=dict)
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dari komponen UI dengan penanganan error.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Returns:
            Dictionary berisi konfigurasi yang diekstrak
        """
        try:
            return extract_downloader_config(ui_components)
        except Exception as e:
            self.log_error(f"Gagal mengekstrak konfigurasi downloader: {str(e)}")
            return self.get_default_config()
    
    @handle_ui_errors(error_component_title="UI Update Error", log_error=True)
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Perbarui komponen UI dari konfigurasi yang dimuat dengan penanganan error.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Dictionary konfigurasi yang akan diterapkan
        """
        if not config or not isinstance(config, dict):
            self.log_warning("Konfigurasi tidak valid untuk pembaruan UI, menggunakan default")
            config = self.get_default_config()
            
        update_downloader_ui(ui_components, config)
    
    @handle_ui_errors(
        error_component_title="Config Load Error",
        log_error=True,
        return_type=dict
    )
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Muat konfigurasi dengan auto-deteksi API key dan fallback ke default.
        
        Untuk handler non-persistent, ini akan selalu mengembalikan konfigurasi default
        atau state konfigurasi di memori saat ini.
        
        Args:
            config_filename: Nama file opsional untuk memuat konfigurasi
            
        Returns:
            Dictionary berisi konfigurasi yang dimuat
        """
        # Untuk handler non-persistent, kembalikan state di memori atau default
        if not self.persistence_enabled:
            if hasattr(self, '_config_state') and self._config_state.data:
                self.log_debug("Menggunakan state konfigurasi di memori untuk handler non-persistent")
                return self._config_state.data
            else:
                self.log_debug("Menggunakan konfigurasi default untuk handler non-persistent")
                return self.get_default_config()
        
        # Untuk handler persistent, muat dari file
        try:
            filename = config_filename or self.config_filename
            
            # Muat dari file
            config = self.config_manager.load_config(filename)
            
            if not config:
                self.log_info(f"File {filename} tidak ditemukan, menggunakan konfigurasi default")
                config = self.get_default_config()
                
                # Simpan default ke file untuk pertama kali
                if self.persistence_enabled:
                    self.config_manager.save_config(config, filename)
                    self.log_info(f"Konfigurasi default disimpan ke {filename}")
            
            # Auto-deteksi dan set API key dari Colab secrets
            config = set_api_key_to_config(config, force_refresh=False)
            
            # Perbarui state di memori
            if hasattr(self, '_config_state'):
                self._config_state.data = config
            
            return config
            
        except Exception as e:
            self.log_error(f"Error saat memuat konfigurasi: {str(e)}")
            return self.get_default_config()
    
    @handle_ui_errors(
        error_component_title="Config Save Error",
        log_error=True,
        return_type=bool
    )
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config with merge strategy for dataset_config.yaml.
        
        For non-persistent handlers, this will only update the in-memory state
        and will not attempt to save to disk.
        
        Args:
            ui_components: Dictionary containing UI components
            config_filename: Optional filename to save config to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ekstrak konfigurasi dari UI
            current_config = self.extract_config(ui_components)
            
            # Validasi sebelum menyimpan
            validation = self.validate_config(current_config)
            if not validation['valid']:
                self.logger.error(f"Validasi konfigurasi gagal: {'; '.join(validation['errors'])}")
                return False
            
            # Gunakan implementasi parent class untuk non-persistent handlers
            if not self.persistence_enabled:
                # Panggil implementasi parent class untuk non-persistent handlers
                return super().save_config(current_config, config_filename)
            
            # Untuk persistent handlers, gunakan merge strategy
            filename = config_filename or self.config_filename
            
            # Muat konfigurasi yang ada untuk digabungkan
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Gabungkan dengan strategi yang aman
            merged_config = self._merge_downloader_config(existing_config, current_config)
            
            # Simpan konfigurasi yang sudah digabungkan
            success = self.config_manager.save_config(merged_config, filename)
            
            # Perbarui state di memori jika tersedia
            if hasattr(self, '_config_state'):
                self._config_state.data = merged_config
            
            if success:
                self.logger.info(f"Konfigurasi berhasil disimpan ke {filename}")
                return True
            else:
                self.logger.error(f"Gagal menyimpan konfigurasi ke {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saat menyimpan konfigurasi: {str(e)}")
            return False
    
    @handle_ui_errors(
        error_component_title="Config Reset Error",
        log_error=True,
        return_type=bool
    )
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset konfigurasi ke default dan perbarui UI.
        
        Untuk handler non-persistent, ini hanya akan mereset state di memori
        dan tidak akan menyimpan ke disk.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config_filename: Nama file opsional untuk menyimpan konfigurasi reset
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            self.log_info("Memulai reset konfigurasi...")
            
            # Simpan API key saat ini dari UI menggunakan helper function
            current_api_key = get_api_key_from_ui(ui_components)
            if current_api_key:
                self.log_info("Menyimpan API key saat ini dari UI")
            
            # Auto-deteksi API key dari berbagai sumber menggunakan helper function
            default_config = self.get_default_config()
            default_config = detect_and_set_api_key(default_config, ui_components)
            
            # Log informasi tentang API key
            if default_config.get('data', {}).get('roboflow', {}).get('api_key'):
                self.log_info("API key disimpan ke konfigurasi default")
            else:
                # Auto-deteksi dari Colab secrets sudah dilakukan di detect_and_set_api_key
                self.log_info("Tidak ada API key yang tersedia")
            
            # Gunakan implementasi parent class untuk non-persistent handlers
            if not self.persistence_enabled:
                # Perbarui UI dengan konfigurasi default
                try:
                    self.update_ui(ui_components, default_config)
                    self.log_info("UI diperbarui dengan konfigurasi default (persistence dinonaktifkan)")
                    # Panggil implementasi parent class untuk non-persistent handlers
                    return super().reset_config(default_config, config_filename)
                except Exception as e:
                    self.log_error(f"Error saat memperbarui UI: {str(e)}")
                    return False
            
            # Perbarui UI dengan konfigurasi default
            try:
                self.update_ui(ui_components, default_config)
                self.log_info("UI diperbarui dengan konfigurasi default")
            except Exception as e:
                self.log_error(f"Error saat memperbarui UI: {str(e)}")
                return False
            
            # Simpan konfigurasi default ke file
            try:
                filename = config_filename or self.config_filename
                success = self.config_manager.save_config(default_config, filename)
                
                # Perbarui state di memori jika tersedia
                if hasattr(self, '_config_state'):
                    self._config_state.data = default_config
                
                if success:
                    self.log_info(f"Konfigurasi default disimpan ke {filename}")
                    return True
                else:
                    self.log_error(f"Gagal menyimpan konfigurasi default ke {filename}")
                    return False
            except Exception as e:
                self.log_error(f"Error saat menyimpan konfigurasi default: {str(e)}")
                return False
                
        except Exception as e:
            self.log_error(f"Error saat reset konfigurasi: {str(e)}")
            return False
    
    @handle_ui_errors(
        error_component_title="Config Validation Error",
        log_error=True,
        return_type=dict
    )
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi konfigurasi dengan pemeriksaan komprehensif untuk downloader.
        
        Args:
            config: Dictionary konfigurasi yang akan divalidasi
            
        Returns:
            Dictionary berisi hasil validasi dengan key 'status' dan 'errors'
        """
        try:
            # Gunakan centralized validation module dari utils
            self.log_info("🔍 Memvalidasi konfigurasi downloader")
            validation = validate_config(config)
            
            # Format validation summary untuk UI
            from smartcash.ui.dataset.downloader.utils.validation_utils import format_validation_summary
            
            # Format untuk log output (plain text)
            summary_text = format_validation_summary(validation, html_format=False)
            
            # Format untuk summary container (HTML)
            summary_html = format_validation_summary(validation, html_format=True)
            
            # Log hasil validasi ke log output
            if validation.get('valid', False):
                self.log_info(summary_text)
            else:
                self.log_error(summary_text)
            
            # Update summary container dengan HTML format
            summary_container = self.ui_components.get('summary_container')
            if summary_container:
                summary_container.clear_output()
                summary_container.append_html(summary_html)
            
            # Ensure 'status' key untuk API consistency
            validation['status'] = validation.get('valid', False)
            
            return validation
            
        except Exception as e:
            self.log_error(f"❌ Error saat memvalidasi konfigurasi: {str(e)}")
            return {
                'status': False,
                'valid': False,
                'errors': [f"Error validasi: {str(e)}"],
                'warnings': []
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
                    
                    # Merge roboflow config menggunakan helper function
                    if 'roboflow' in new_downloader['data']:
                        if 'api_key' in new_downloader['data']['roboflow']:
                            merged = update_config_with_api_key(merged, new_downloader['data']['roboflow']['api_key'])
                        else:
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
            self.log_error(f"❌ Error saat mengekstrak konfigurasi dari UI: {str(e)}")
            return self.get_default_config()
    
    def get_api_key_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Dapatkan status API key dengan info auto-detection"""
        try:
            from smartcash.ui.dataset.downloader.utils.colab_secrets import validate_api_key
            
            # Dapatkan API key dari berbagai sumber
            detected_key = get_api_key_from_secrets()
            current_key = get_api_key_from_ui(ui_components)
            
            # Tentukan sumber dan key yang akan digunakan
            source = 'not_provided'
            key = ''
            
            if detected_key:
                source = 'colab_secret'
                key = detected_key
            elif current_key:
                source = 'manual_input'
                key = current_key
                
            # Validasi key jika tersedia
            if key:
                validation = validate_api_key(key)
                valid = validation['valid']
                message = f"{'Auto-deteksi dari Colab' if source == 'colab_secret' else 'Input manual'}: {validation['message']}"
                key_preview = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else '****'
            else:
                valid = False
                message = 'API key belum diisi'
                key_preview = '****'
                
            # Return hasil dengan format yang konsisten
            return {
                'source': source,
                'valid': valid,
                'status': valid,  # Untuk API consistency
                'message': message,
                'key_preview': key_preview
            }
                
        except Exception as e:
            self.log_error(f"Error saat memeriksa API key: {str(e)}")
            return {
                'source': 'error',
                'valid': False,
                'status': False,
                'message': f"Error saat memeriksa API key: {str(e)}",
                'key_preview': '****'
            }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default dengan workers optimal dan struktur yang tepat"""
        try:
            # Import sudah ada di level modul, tidak perlu import lagi
            default_config = get_default_downloader_config()
            
            # Perbarui dengan workers optimal
            default_config['download']['max_workers'] = get_download_workers()
            default_config['uuid_renaming']['parallel_workers'] = get_rename_workers(5000)
            
            return default_config
            
        except Exception as e:
            self.log_error(f"❌ Error saat mendapatkan konfigurasi default: {str(e)}")
            raise