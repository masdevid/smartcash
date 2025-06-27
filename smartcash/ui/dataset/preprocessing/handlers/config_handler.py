"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Config handler dengan konsistensi pattern dan DRY utilities
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Preprocessing config handler dengan consistent pattern dan error handling"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
        self._ui_components: Optional[Dict[str, Any]] = None
    
    @property
    def logger_bridge(self):
        """Get logger bridge dari UI components atau fallback ke parent logger"""
        if self._ui_components and 'logger_bridge' in self._ui_components:
            return self._ui_components['logger_bridge']
        return self.logger
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI dengan enhanced error handling"""
        try:
            config = extract_preprocessing_config(ui_components)
            self._log_operation_success("Config extracted successfully")
            return config
        except Exception as e:
            error_msg = f"Error extracting config: {str(e)}"
            self._log_operation_error(error_msg)
            return self.get_default_config()
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan config dan error handling"""
        try:
            update_preprocessing_ui(ui_components, config)
            self._log_operation_success("UI updated with config")
        except Exception as e:
            error_msg = f"Error updating UI: {str(e)}"
            self._log_operation_error(error_msg)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback handling"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            config = get_default_preprocessing_config()
            self._log_operation_info("Default config loaded")
            return config
        except Exception as e:
            error_msg = f"Error loading defaults: {str(e)}"
            self._log_operation_error(error_msg)
            return self._get_minimal_fallback_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan inheritance dan enhanced error handling"""
        filename = config_filename or self.config_filename
        
        try:
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_operation_warning("Config kosong, menggunakan default")
                return self.get_default_config()
            
            # Handle inheritance dengan _base_
            if '_base_' in config:
                config = self._resolve_inheritance(config, filename)
            
            self._log_operation_success(f"Config loaded dari {filename}")
            return config
            
        except Exception as e:
            error_msg = f"Error loading config: {str(e)}"
            self._log_operation_error(error_msg)
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan proper lifecycle management"""
        filename = config_filename or self.config_filename
        
        try:
            # Extract current UI config
            ui_config = self.extract_config(ui_components)
            
            # Save config
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_operation_success(f"Config saved to {filename}")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                self._log_operation_error("Failed to save config")
                return False
                
        except Exception as e:
            error_msg = f"Error saving config: {str(e)}"
            self._log_operation_error(error_msg)
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset config ke defaults dengan proper UI update"""
        filename = config_filename or self.config_filename
        
        try:
            default_config = self.get_default_config()
            
            # Save default config
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_operation_success("Config reset to defaults")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self._log_operation_error("Failed to reset config")
                return False
                
        except Exception as e:
            error_msg = f"Error resetting config: {str(e)}"
            self._log_operation_error(error_msg)
            return False
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components untuk access ke logger bridge"""
        self._ui_components = ui_components
    
    def _resolve_inheritance(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Resolve config inheritance dengan _base_ support"""
        try:
            base_config_name = config.pop('_base_')
            base_config = self.config_manager.load_config(base_config_name) or {}
            
            merged_config = self._deep_merge(base_config, config)
            self._log_operation_info(f"Resolved inheritance for {config_name} with base {base_config_name}")
            return merged_config
            
        except Exception as e:
            error_msg = f"Error resolving inheritance for {config_name}: {str(e)}"
            self._log_operation_error(error_msg)
            return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge untuk config inheritance"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str) -> None:
        """Refresh UI setelah save dengan error handling"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self._log_operation_info("UI refreshed with saved config")
        except Exception as e:
            self._log_operation_warning(f"Error refreshing UI: {str(e)}")
    
    def _get_minimal_fallback_config(self) -> Dict[str, Any]:
        """Minimal fallback config jika semua load gagal"""
        return {
            'preprocessing': {
                'enabled': True,
                'target_splits': ['train', 'valid'],
                'normalization': {'enabled': True, 'target_size': [640, 640]}
            },
            'performance': {'batch_size': 32}
        }
    
    # === CONSISTENT LOGGING METHODS ===
    
    def _log_operation_success(self, message: str) -> None:
        """Log success operation"""
        if hasattr(self.logger_bridge, 'success'):
            self.logger_bridge.success(message)
        else:
            self.logger_bridge.info(f"✅ {message}")
    
    def _log_operation_info(self, message: str) -> None:
        """Log info operation"""
        self.logger_bridge.info(message)
    
    def _log_operation_warning(self, message: str) -> None:
        """Log warning operation"""
        self.logger_bridge.warning(message)
    
    def _log_operation_error(self, message: str) -> None:
        """Log error operation"""
        self.logger_bridge.error(message)