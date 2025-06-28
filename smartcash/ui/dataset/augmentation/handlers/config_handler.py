"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler dengan backend integration dan proper inheritance
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_updater import update_augmentation_ui
from smartcash.common.config.manager import get_config_manager

class AugmentationConfigHandler(ConfigHandler):
    """Config handler dengan backend integration dan proper logging"""
    
    def __init__(self, module_name: str = 'augmentation', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'augmentation_config.yaml'
        self._ui_components = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan backend compatibility"""
        extracted = extract_augmentation_config(ui_components)
        
        # Add backend-specific configs
        extracted['backend'] = {
            'service_enabled': True,
            'progress_tracking': True,
            'async_processing': False
        }
        
        return extracted
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan inheritance handling"""
        update_augmentation_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dari augmentation_config.yaml"""
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        return get_default_augmentation_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load dengan inheritance dari base_config.yaml"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_to_ui("⚠️ Config kosong, menggunakan default", "warning")
                return self.get_default_config()
                
            # Ensure required sections exist
            if 'augmentation' not in config:
                config['augmentation'] = {}
                
            # Ensure basic structure exists
            if 'basic' not in config['augmentation']:
                config['augmentation']['basic'] = self.get_default_config().get('basic', {})
                
            # Ensure advanced structure exists
            if 'advanced' not in config['augmentation']:
                config['augmentation']['advanced'] = self.get_default_config().get('advanced', {})
                return self.get_default_config()
            
            # Handle inheritance dari _base_
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                self._log_to_ui(f"📂 Config loaded dengan inheritance dari {filename}", "info")
                return merged_config
            
            self._log_to_ui(f"📂 Config loaded dari {filename}", "info")
            return config
            
        except Exception as e:
            self._log_to_ui(f"❌ Error loading config: {str(e)}", "error")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save dengan auto refresh dan backend update"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            # Validate backend compatibility
            if not self._validate_backend_config(ui_config):
                self._log_to_ui("⚠️ Config disesuaikan untuk kompatibilitas backend", "warning")
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_to_ui(f"✅ Config tersimpan ke {filename}", "success")
                self._refresh_ui_after_save(ui_components, filename)
                self._notify_backend_config_change(ui_components, ui_config)
                return True
            else:
                self._log_to_ui(f"❌ Gagal simpan config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error save config: {str(e)}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset dengan backend notification"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui("🔄 Config direset ke default", "success")
                self.update_ui(ui_components, default_config)
                self._notify_backend_config_change(ui_components, default_config)
                return True
            else:
                self._log_to_ui("❌ Gagal reset config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error reset config: {str(e)}", "error")
            return False
    
    def _validate_backend_config(self, config: Dict[str, Any]) -> bool:
        """Validate config untuk backend compatibility"""
        aug_config = config.get('augmentation', {})
        
        # Ensure required fields
        required_fields = ['num_variations', 'target_count', 'types']
        for field in required_fields:
            if field not in aug_config:
                aug_config[field] = self.get_default_config()['augmentation'][field]
        
        # Validate ranges
        aug_config['num_variations'] = max(1, min(10, aug_config.get('num_variations', 3)))
        aug_config['target_count'] = max(100, min(2000, aug_config.get('target_count', 500)))
        
        return True
    
    def _notify_backend_config_change(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Notify backend tentang config changes"""
        try:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(ui_components, "🔄 Backend config updated", "info")
        except Exception:
            pass  # Silent fail
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configs dengan deep merge"""
        import copy
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key == '_base_':
                continue
                
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge helper"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str):
        """Auto refresh UI setelah save"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self._log_to_ui("🔄 UI direfresh dengan config tersimpan", "info")
        except Exception as e:
            self._log_to_ui(f"⚠️ Error refresh UI: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """Log menggunakan UI utilities"""
        try:
            from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
            log_to_ui(self._ui_components, message, level)
        except Exception:
            print(f"[{level.upper()}] {message}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components untuk logging"""
        self._ui_components = ui_components