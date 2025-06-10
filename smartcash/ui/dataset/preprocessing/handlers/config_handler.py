"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Fixed config handler dengan Progress Bridge callback integration dan enhanced backend service sync
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Fixed config handler dengan Progress Bridge logging dan backend service integration"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
        self._progress_callback = None
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config tanpa progress feedback"""
        try:
            config = extract_preprocessing_config(ui_components)
            return config
        except Exception as e:
            self._log_to_ui(f"❌ Error extracting config: {str(e)}", "error")
            return self.get_default_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI tanpa progress feedback"""
        try:
            update_preprocessing_ui(ui_components, config)
            self._log_to_ui("🔄 UI components updated dengan konfigurasi terbaru", "success")
        except Exception as e:
            self._log_to_ui(f"❌ Error updating UI: {str(e)}", "error")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config tanpa progress feedback"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            config = get_default_preprocessing_config()
            return config
        except Exception as e:
            self._log_to_ui(f"❌ Error loading default config: {str(e)}", "error")
            return {'preprocessing': {'enabled': True}, 'performance': {'batch_size': 32}}
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config tanpa progress tracking"""
        try:
            filename = config_filename or self.config_filename
            
            # Load preprocessing_config.yaml
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_to_ui("⚠️ Config kosong, menggunakan default", "warning")
                return self.get_default_config()
            
            # Handle inheritance dari _base_
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs_with_validation(base_config, config)
                self._log_to_ui(f"📂 Config loaded dari {filename} dengan inheritance", "success")
                
                # Backend service validation
                self._validate_with_backend_service(merged_config)
                return merged_config
            
            # Direct config tanpa inheritance
            self._validate_with_backend_service(config)
            self._log_to_ui(f"📂 Config loaded dari {filename}", "success")
            return config
            
        except Exception as e:
            self._log_to_ui(f"❌ Error loading config: {str(e)}", "error")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Enhanced save dengan backend validation dan auto-refresh"""
        try:
            filename = config_filename or self.config_filename
            
            # Extract config
            ui_config = self.extract_config(ui_components)
            
            # Backend validation sebelum save
            if not self._validate_with_backend_service(ui_config):
                self._log_to_ui("❌ Konfigurasi tidak valid untuk backend service", "error")
                return False
            
            # Save configuration
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_to_ui(f"✅ Config tersimpan ke {filename}", "success")
                
                # Enhanced refresh dengan backend sync
                self._refresh_ui_with_backend_sync(ui_components, filename)
                return True
            else:
                self._log_to_ui(f"❌ Gagal simpan config ke {filename}", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error save config: {str(e)}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Enhanced reset dengan backend sync tanpa progress tracking"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            # Backend validation
            if not self._validate_with_backend_service(default_config):
                self._log_to_ui("❌ Default config tidak valid", "error")
                return False
            
            # Save default config
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui(f"🔄 Config direset ke default", "success")
                
                # Direct update UI dengan default config dan backend sync
                self.update_ui(ui_components, default_config)
                self._sync_with_backend_service(ui_components, default_config)
                
                return True
            else:
                self._log_to_ui(f"❌ Gagal reset config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error reset config: {str(e)}", "error")
            return False
    
    def _merge_configs_with_validation(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced merge dengan validation"""
        import copy
        
        try:
            merged = copy.deepcopy(base_config)
            
            # Merge dengan validation
            for key, value in override_config.items():
                if key == '_base_':
                    continue
                    
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._deep_merge_with_validation(merged[key], value)
                else:
                    merged[key] = value
            
            # Post-merge validation
            if not self._validate_merged_config(merged):
                self._log_to_ui("⚠️ Merged config memiliki masalah, using fallback", "warning")
                return self.get_default_config()
            
            return merged
            
        except Exception as e:
            self._log_to_ui(f"❌ Error merging configs: {str(e)}", "error")
            return self.get_default_config()
    
    def _deep_merge_with_validation(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dengan validation"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_with_validation(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_merged_config(self, config: Dict[str, Any]) -> bool:
        """Validate merged configuration structure"""
        required_sections = ['preprocessing', 'performance']
        return all(section in config for section in required_sections)
    
    def _validate_with_backend_service(self, config: Dict[str, Any]) -> bool:
        """Validate config dengan backend service"""
        try:
            # Import backend validation
            from smartcash.dataset.preprocessor.utils.config_validator import validate_preprocessing_config
            
            validated = validate_preprocessing_config(config)
            if 'error' in validated:
                self._log_to_ui(f"❌ Backend validation failed: {validated['error']}", "error")
                return False
            
            return True
            
        except Exception as e:
            self._log_to_ui(f"⚠️ Backend validation unavailable: {str(e)}", "warning")
            return True  # Allow if backend validation tidak tersedia
    
    def _refresh_ui_with_backend_sync(self, ui_components: Dict[str, Any], filename: str):
        """Enhanced refresh dengan backend service sync"""
        try:
            # Reload dari file dengan inheritance handling
            saved_config = self.load_config(filename)
            
            if saved_config:
                # Update UI dengan config yang direload
                self.update_ui(ui_components, saved_config)
                
                # Sync dengan backend service
                self._sync_with_backend_service(ui_components, saved_config)
                
                self._log_to_ui("🔄 UI dan backend service sinkron dengan config tersimpan", "success")
            
        except Exception as e:
            self._log_to_ui(f"⚠️ Error refresh UI: {str(e)}", "warning")
    
    def _sync_with_backend_service(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Sync configuration dengan backend preprocessing service"""
        try:
            # Update backend service jika ada
            if 'backend_service' in ui_components:
                backend_service = ui_components['backend_service']
                if hasattr(backend_service, 'update_config'):
                    backend_service.update_config(config)
                    self._log_to_ui("🔄 Backend service config updated", "info")
            
            # Update progress callback reference
            if hasattr(self, '_progress_callback') and self._progress_callback:
                ui_components['progress_callback'] = self._progress_callback
                
        except Exception as e:
            self._log_to_ui(f"⚠️ Backend sync warning: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """Enhanced logging dengan Progress Bridge integration"""
        try:
            # Try UI logger dulu dengan Progress Bridge
            ui_components = getattr(self, '_ui_components', {})
            
            # Progress Bridge logging
            if self._progress_callback:
                try:
                    # Map level ke progress indication
                    progress_level = "current" if level in ["info", "success"] else "overall"
                    self._progress_callback(progress_level, 100, 100, message)
                except Exception:
                    pass
            
            # Standard UI logging
            logger = ui_components.get('logger')
            if logger and hasattr(logger, level):
                log_method = getattr(logger, level)
                log_method(message)
                return
            
            # Fallback ke log_to_accordion
            from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
            log_to_accordion(ui_components, message, level)
                
        except Exception:
            # Final fallback
            print(f"[{level.upper()}] {message}")
    
    def _update_progress_safe(self, level: str, current: int, total: int, message: str):
        """Safe progress update dengan Progress Bridge"""
        try:
            if self._progress_callback:
                self._progress_callback(level, current, total, message)
        except Exception:
            pass  # Silent fail untuk prevent breaking main process
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Enhanced set UI components dengan Progress Bridge registration"""
        self._ui_components = ui_components
        
        # Register progress callback dari UI components
        if 'progress_callback' in ui_components:
            self._progress_callback = ui_components['progress_callback']
    
    def set_progress_callback(self, callback):
        """Set Progress Bridge callback untuk backend integration"""
        self._progress_callback = callback