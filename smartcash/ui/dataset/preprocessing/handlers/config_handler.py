"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Fixed config handler dengan proper logging ke UI dan load config yang benar
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Fixed config handler dengan proper UI logging dan load"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dengan DRY approach"""
        return extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_preprocessing_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dari defaults.py"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan proper inheritance handling"""
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
                merged_config = self._merge_configs(base_config, config)
                self._log_to_ui(f"📂 Config loaded dari {filename} dengan inheritance", "info")
                return merged_config
            
            self._log_to_ui(f"📂 Config loaded dari {filename}", "info")
            return config
            
        except Exception as e:
            self._log_to_ui(f"❌ Error loading config: {str(e)}", "error")
            return self.get_default_config()
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base config dengan override config"""
        import copy
        
        merged = copy.deepcopy(base_config)
        
        # Override sections yang ada di preprocessing_config.yaml
        for key, value in override_config.items():
            if key == '_base_':
                continue
                
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge untuk nested dicts
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save dengan proper UI logging"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_to_ui(f"✅ Config tersimpan ke {filename}", "success")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                self._log_to_ui(f"❌ Gagal simpan config ke {filename}", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error save config: {str(e)}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset dengan proper UI logging"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui(f"🔄 Config direset ke default", "success")
                # Direct update UI dengan default config
                self.update_ui(ui_components, default_config)
                return True
            else:
                self._log_to_ui(f"❌ Gagal reset config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error reset config: {str(e)}", "error")
            return False
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str):
        """Refresh UI dengan reload dari file"""
        try:
            # Reload dari file dengan inheritance handling
            saved_config = self.load_config(filename)
            
            if saved_config:
                # Update UI dengan config yang direload
                self.update_ui(ui_components, saved_config)
                self._log_to_ui("🔄 UI direfresh dengan config tersimpan", "info")
            
        except Exception as e:
            self._log_to_ui(f"⚠️ Error refresh UI: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """Log langsung ke UI components"""
        try:
            # Coba log ke UI logger dulu
            ui_components = getattr(self, '_ui_components', {})
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
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk logging"""
        self._ui_components = ui_components