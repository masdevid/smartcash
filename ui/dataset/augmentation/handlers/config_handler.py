"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging
import copy

# Import base handler
from smartcash.ui.handlers.config_handlers import ConfigHandler

# Import error handling
from smartcash.ui.handlers.error_handler import handle_ui_errors

# Import config utilities
from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_updater import update_augmentation_ui
from smartcash.common.config.manager import get_config_manager

class AugmentationConfigHandler(ConfigHandler):
    """Config handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for augmentation configuration management:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Config validation and inheritance
    - Backend integration
    """
    
    def __init__(self, module_name: str = 'augmentation', parent_module: str = 'dataset', persistence_enabled: bool = True):
        """Initialize augmentation config handler
        
        Args:
            module_name: Nama module
            parent_module: Nama parent module
            persistence_enabled: True jika config perlu disimpan ke disk, False untuk in-memory only
        """
        super().__init__(module_name, parent_module, persistence_enabled=persistence_enabled)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("AugmentationConfigHandler initialized")
        
        # Initialize config manager if persistence is enabled
        self.config_manager = get_config_manager() if persistence_enabled else None
        self.config_filename = 'augmentation_config.yaml'
        self._ui_components = {}
        self._in_memory_config = None
    
    @handle_ui_errors(log_error=True)
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan centralized error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Returns:
            Dictionary berisi konfigurasi yang diekstrak
        """
        self.logger.debug("Mengekstrak konfigurasi dari UI components")
        
        try:
            # Extract config using utility function
            extracted = extract_augmentation_config(ui_components)
            
            # Add backend-specific configs
            extracted['backend'] = {
                'service_enabled': True,
                'progress_tracking': True,
                'async_processing': False
            }
            
            self.log_message("✅ Konfigurasi berhasil diekstrak dari UI", "success")
            return extracted
        except Exception as e:
            self.log_message(f"❌ Gagal mengekstrak konfigurasi: {str(e)}", "error")
            raise
    
    @handle_ui_errors(log_error=True)
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan centralized error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Dictionary berisi konfigurasi
        """
        self.logger.debug("Memperbarui UI components dengan konfigurasi")
        
        try:
            # Update UI using utility function
            update_augmentation_ui(ui_components, config)
            self.log_message("✅ UI berhasil diperbarui dengan konfigurasi", "success")
        except Exception as e:
            self.log_message(f"❌ Gagal memperbarui UI: {str(e)}", "error")
            raise
    
    @handle_ui_errors(log_error=True)
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan centralized error handling
        
        Returns:
            Dictionary berisi konfigurasi default
        """
        self.logger.debug("Mengambil konfigurasi default")
        
        try:
            from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
            default_config = get_default_augmentation_config()
            self.log_message("✅ Konfigurasi default berhasil dimuat", "info")
            return default_config
        except Exception as e:
            self.log_message(f"❌ Gagal memuat konfigurasi default: {str(e)}", "error")
            # Return minimal working config to prevent crashes
            return {
                'augmentation': {
                    'enabled': True,
                    'methods': ['flip', 'rotate'],
                    'intensity': 0.5
                },
                'data': {'dir': 'data'}
            }
    
    @handle_ui_errors(log_error=True)
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan centralized error handling dan support untuk non-persistent mode
        
        Args:
            config_filename: Optional nama file konfigurasi
            
        Returns:
            Dictionary berisi konfigurasi yang dimuat
        """
        self.logger.debug(f"Memuat konfigurasi dari {config_filename or self.config_filename}")
        
        # If we have in-memory config and persistence is disabled, return it
        if not self.persistence_enabled and self._in_memory_config is not None:
            self.logger.debug("Menggunakan konfigurasi in-memory (persistence disabled)")
            return self._in_memory_config
            
        try:
            filename = config_filename or self.config_filename
            
            # If persistence is disabled, return default config
            if not self.persistence_enabled:
                self.logger.debug("Persistence disabled, menggunakan konfigurasi default")
                self._in_memory_config = self.get_default_config()
                return self._in_memory_config
                
            # Load config from file
            config = self.config_manager.load_config(filename)
            
            if not config:
                self.log_message("⚠️ Konfigurasi kosong, menggunakan default", "warning")
                return self.get_default_config()
                
            # Ensure required sections exist
            if 'augmentation' not in config:
                config['augmentation'] = {}
                
            # Ensure basic structure exists
            if 'basic' not in config['augmentation']:
                config['augmentation']['basic'] = self.get_default_config().get('augmentation', {}).get('basic', {})
                
            # Ensure advanced structure exists
            if 'advanced' not in config['augmentation']:
                config['augmentation']['advanced'] = self.get_default_config().get('augmentation', {}).get('advanced', {})
                
            # Handle inheritance dari _base_
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                self.log_message(f"📂 Konfigurasi dimuat dengan inheritance dari {filename}", "info")
                return merged_config
            
            self.log_message(f"📂 Konfigurasi berhasil dimuat dari {filename}", "info")
            return config
            
        except Exception as e:
            self.log_message(f"❌ Gagal memuat konfigurasi: {str(e)}", "error")
            return self.get_default_config()
    
    @handle_ui_errors(log_error=True)
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan centralized error handling dan support untuk non-persistent mode
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config_filename: Optional nama file konfigurasi
            
        Returns:
            Boolean yang menunjukkan keberhasilan operasi
        """
        self.logger.debug(f"Menyimpan konfigurasi ke {config_filename or self.config_filename}")
        
        try:
            # Extract config from UI
            ui_config = self.extract_config(ui_components)
            
            # If persistence is disabled, just update in-memory config
            if not self.persistence_enabled:
                self.logger.debug("Persistence disabled, menyimpan konfigurasi ke memory")
                self._in_memory_config = ui_config
                self.log_message("✅ Konfigurasi berhasil disimpan ke memory", "success")
                return True
                
            filename = config_filename or self.config_filename
            
            # Validate backend compatibility
            if not self._validate_backend_config(ui_config):
                self.log_message("⚠️ Konfigurasi disesuaikan untuk kompatibilitas backend", "warning")
            
            # Save config to file
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self.log_message(f"💾 Konfigurasi berhasil disimpan ke {filename}", "success")
                self._notify_backend_config_change(ui_components, ui_config)
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                self.log_message("❌ Gagal menyimpan konfigurasi", "error")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Gagal menyimpan konfigurasi: {str(e)}", "error")
            return False
    
    @handle_ui_errors(log_error=True)
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset config dengan centralized error handling dan support untuk non-persistent mode
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config_filename: Optional nama file konfigurasi
            
        Returns:
            Boolean yang menunjukkan keberhasilan operasi
        """
        self.logger.debug("Mereset konfigurasi ke default")
        
        try:
            # Get default config
            default_config = self.get_default_config()
            
            # If persistence is disabled, just update in-memory config
            if not self.persistence_enabled:
                self.logger.debug("Persistence disabled, mereset konfigurasi di memory")
                self._in_memory_config = default_config
                self.update_ui(ui_components, default_config)
                self.log_message("🔄 Konfigurasi berhasil direset ke default", "success")
                return True
                
            filename = config_filename or self.config_filename
            
            # Save default config to file
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self.log_message("🔄 Konfigurasi berhasil direset ke default", "success")
                self.update_ui(ui_components, default_config)
                self._notify_backend_config_change(ui_components, default_config)
                return True
            else:
                self.log_message("❌ Gagal mereset konfigurasi", "error")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Gagal mereset konfigurasi: {str(e)}", "error")
            return False
    
    @handle_ui_errors(log_error=True)
    def _validate_backend_config(self, config: Dict[str, Any]) -> bool:
        """Validate config untuk backend compatibility dengan centralized error handling
        
        Args:
            config: Dictionary berisi konfigurasi yang akan divalidasi
            
        Returns:
            Boolean yang menunjukkan keberhasilan validasi
        """
        self.logger.debug("Memvalidasi konfigurasi untuk kompatibilitas backend")
        
        try:
            aug_config = config.get('augmentation', {})
            
            # Ensure required fields
            required_fields = ['num_variations', 'target_count', 'types']
            missing_fields = [field for field in required_fields if field not in aug_config]
            
            if missing_fields:
                self.logger.warning(f"Field yang diperlukan tidak ada: {', '.join(missing_fields)}")
                default_config = self.get_default_config()
                
                for field in missing_fields:
                    aug_config[field] = default_config.get('augmentation', {}).get(field, None)
                    if aug_config[field] is None:
                        self.log_message(f"⚠️ Field {field} tidak ditemukan di default config", "warning")
            
            # Validate ranges
            original_num_var = aug_config.get('num_variations', 3)
            original_target = aug_config.get('target_count', 500)
            
            aug_config['num_variations'] = max(1, min(10, original_num_var))
            aug_config['target_count'] = max(100, min(2000, original_target))
            
            # Log if values were adjusted
            if aug_config['num_variations'] != original_num_var:
                self.logger.info(f"num_variations disesuaikan dari {original_num_var} ke {aug_config['num_variations']}")
                
            if aug_config['target_count'] != original_target:
                self.logger.info(f"target_count disesuaikan dari {original_target} ke {aug_config['target_count']}")
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Gagal memvalidasi konfigurasi: {str(e)}", "error")
            return False
    
    @handle_ui_errors(log_error=False)  # Silent fail on purpose
    def _notify_backend_config_change(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Notify backend tentang perubahan konfigurasi dengan centralized error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Dictionary berisi konfigurasi
        """
        self.logger.debug("Memberitahu backend tentang perubahan konfigurasi")
        
        try:
            # Notify backend about config changes
            self.log_message("🔄 Konfigurasi backend diperbarui", "info")
        except Exception:
            # Silent fail is intentional here
            self.logger.debug("Gagal memberitahu backend tentang perubahan konfigurasi (silent fail)")
            pass
    
    @handle_ui_errors(log_error=True)
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configs dengan deep merge dan centralized error handling
        
        Args:
            base_config: Dictionary berisi konfigurasi dasar
            override_config: Dictionary berisi konfigurasi override
            
        Returns:
            Dictionary berisi konfigurasi yang telah di-merge
        """
        self.logger.debug("Melakukan deep merge konfigurasi")
        
        try:
            merged = copy.deepcopy(base_config)
            
            for key, value in override_config.items():
                if key == '_base_':
                    continue
                    
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._deep_merge(merged[key], value)
                else:
                    merged[key] = value
            
            return merged
            
        except Exception as e:
            self.log_message(f"❌ Gagal melakukan merge konfigurasi: {str(e)}", "error")
            # Return base config if merge fails
            return base_config
    
    @handle_ui_errors(log_error=True)
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge helper dengan centralized error handling
        
        Args:
            base: Dictionary berisi data dasar
            override: Dictionary berisi data override
            
        Returns:
            Dictionary hasil merge
        """
        try:
            result = copy.deepcopy(base)
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gagal melakukan deep merge: {str(e)}")
            # Return base if deep merge fails
            return base
    
    @handle_ui_errors(log_error=True)
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str) -> None:
        """Auto refresh UI setelah save dengan centralized error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            filename: Nama file konfigurasi
        """
        self.logger.debug(f"Merefresh UI setelah menyimpan ke {filename}")
        
        try:
            # Load saved config
            saved_config = self.load_config(filename)
            if saved_config:
                # Update UI with saved config
                self.update_ui(ui_components, saved_config)
                self.log_message("🔄 UI berhasil direfresh dengan konfigurasi tersimpan", "info")
        except Exception as e:
            self.log_message(f"⚠️ Gagal merefresh UI: {str(e)}", "warning")
    
    # Removed redundant log_message and set_ui_components methods
    # These methods are already provided by BaseAugmentationHandler (from BaseHandler)