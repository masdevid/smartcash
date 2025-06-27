"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Simplified config handler tanpa complexity yang berlebihan
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.logger_bridge import UILoggerBridge

class PreprocessingConfigHandler(ConfigHandler):
    """Simplified preprocessing config handler"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
        self.logger_bridge: Optional[UILoggerBridge] = None
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI"""
        try:
            return extract_preprocessing_config(ui_components)
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error extracting config: {str(e)}")
            return self.get_default_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan config"""
        try:
            update_preprocessing_ui(ui_components, config)
            if self.logger_bridge:
                self.logger_bridge.success("UI updated with config")
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error updating UI: {str(e)}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error loading defaults: {str(e)}")
            return {'preprocessing': {'enabled': True}, 'performance': {'batch_size': 32}}
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan inheritance handling"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                if self.logger_bridge:
                    self.logger_bridge.warning("Config kosong, menggunakan default")
                return self.get_default_config()
            
            # Handle inheritance
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                if self.logger_bridge:
                    self.logger_bridge.success(f"Config loaded dari {filename} dengan inheritance")
                return merged_config
            
            if self.logger_bridge:
                self.logger_bridge.success(f"Config loaded dari {filename}")
            return config
            
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error loading config: {str(e)}")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config tanpa complex progress tracking"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                if self.logger_bridge:
                    self.logger_bridge.success(f"Config saved to {filename}")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                if self.logger_bridge:
                    self.logger_bridge.error("Failed to save config")
                return False
                
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error saving config: {str(e)}")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset config ke defaults"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                if self.logger_bridge:
                    self.logger_bridge.success("Config reset to defaults")
                self.update_ui(ui_components, default_config)
                return True
            else:
                if self.logger_bridge:
                    self.logger_bridge.error("Failed to reset config")
                return False
                
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"Error resetting config: {str(e)}")
            return False
    
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
        """Refresh UI setelah save"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                if self.logger_bridge:
                    self.logger_bridge.info("UI refreshed with saved config")
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.warning(f"Error refreshing UI: {str(e)}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components dan logger bridge
        
        Args:
            ui_components: Dictionary berisi komponen UI, termasuk logger_bridge
        """
        self._ui_components = ui_components
        self.logger_bridge = ui_components.get('logger_bridge')