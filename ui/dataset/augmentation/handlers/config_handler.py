"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Config handler dengan proper inheritance dan UI logging yang diperbaiki
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_updater import update_augmentation_ui
from smartcash.common.config.manager import get_config_manager

class AugmentationConfigHandler(ConfigHandler):
    """Config handler dengan proper UI logging dan inheritance support"""
    
    def __init__(self, module_name: str = 'augmentation', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'augmentation_config.yaml'
        self._ui_components = {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dengan DRY approach"""
        return extract_augmentation_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan inheritance handling"""
        update_augmentation_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dari defaults.py"""
        from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
        return get_default_augmentation_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """CRITICAL: Load dengan inheritance handling"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_to_ui("⚠️ Config kosong, menggunakan default", "warning")
                return self.get_default_config()
            
            # CRITICAL: Handle inheritance dari _base_
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
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """CRITICAL: Save dengan auto refresh"""
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
        """CRITICAL: Reset dengan auto refresh"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui("🔄 Config direset ke default", "success")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self._log_to_ui("❌ Gagal reset config", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Error reset config: {str(e)}", "error")
            return False
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """CRITICAL: Merge base config dengan override"""
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
        """Deep merge dictionaries"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str):
        """CRITICAL: Auto refresh UI setelah save"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self._log_to_ui("🔄 UI direfresh dengan config tersimpan", "info")
        except Exception as e:
            self._log_to_ui(f"⚠️ Error refresh UI: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """CRITICAL: Log ke UI components dengan fallback"""
        try:
            ui_components = self._ui_components
            logger = ui_components.get('logger')
            
            if logger and hasattr(logger, level):
                log_method = getattr(logger, level)
                log_method(message)
                return
            
            # Fallback ke log widget
            widget = ui_components.get('log_output') or ui_components.get('status')
            if widget and hasattr(widget, 'clear_output'):
                from IPython.display import display, HTML
                color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
                color = color_map.get(level, '#007bff')
                html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
                
                with widget:
                    display(HTML(html))
                return
                
        except Exception:
            print(f"[{level.upper()}] {message}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """CRITICAL: Set UI components untuk logging"""
        self._ui_components = ui_components