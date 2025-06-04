"""
File: smartcash/ui/handlers/config_handlers.py
Deskripsi: Fixed ConfigHandler dengan proper generator cleanup dan logger integration
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe

class ConfigHandler(ABC):
    """Fixed ConfigHandler dengan automatic generator cleanup dan enhanced logging"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.logger = get_logger(f"smartcash.ui.{parent_module}.{module_name}" if parent_module else f"smartcash.ui.{module_name}")
        self.config_manager = get_config_manager()
        self.callbacks = []
        self._current_config = {}
        
    @abstractmethod
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dari komponen UI"""
        pass
        
    @abstractmethod
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari konfigurasi yang dimuat"""
        pass
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback handling"""
        try:
            module_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.defaults" if self.parent_module else f"smartcash.ui.{self.module_name}.defaults"
            module = __import__(module_path, fromlist=['DEFAULT_CONFIG'])
            default_config = getattr(module, 'DEFAULT_CONFIG', {})
            
            if default_config:
                self.logger.info(f"📋 Loaded defaults for {self.module_name}")
                return default_config
            
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"🔍 No defaults found for {self.module_name}: {str(e)}")
        
        return {
            'module_name': self.module_name,
            'version': '1.0.0',
            'created_by': 'SmartCash',
            'settings': {}
        }
    
    def _force_close_generators(self, ui_components: Dict[str, Any]) -> int:
        """Force close all generators untuk prevent RuntimeError - one-liner batch close"""
        try:
            generators = [v for v in ui_components.values() if hasattr(v, 'close') and hasattr(v, '__next__')]
            [gen.close() for gen in generators]
            count = len(generators)
            count > 0 and self.logger.info(f"🧹 Closed {count} generators to prevent RuntimeError")
            return count
        except Exception as e:
            self.logger.warning(f"⚠️ Generator cleanup error: {str(e)}")
            return 0
    
    def save_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Save config dengan comprehensive error handling dan generator cleanup"""
        try:
            self.logger.info(f"💾 Starting save config for {self.module_name}...")
            
            # Force close generators first
            generator_count = self._force_close_generators(ui_components)
            
            # Before save hook
            self.before_save(ui_components)
            
            # Extract config dengan error handling
            try:
                config = self.extract_config(ui_components)
                self.logger.debug(f"📊 Extracted config with {len(config)} keys")
            except Exception as e:
                self.logger.error(f"💥 Config extraction failed: {str(e)}")
                self.after_save_failure(ui_components, f"Extraction error: {str(e)}")
                return False
            
            # Save config dengan error handling
            try:
                success = self.config_manager.save_config(config, config_name or self.module_name)
                if not success:
                    raise Exception("Config manager returned False")
                    
                self.logger.success(f"✅ Config saved successfully")
            except Exception as e:
                self.logger.error(f"💥 Config save failed: {str(e)}")
                self.after_save_failure(ui_components, f"Save error: {str(e)}")
                return False
            
            # Success handling
            self._current_config = config
            ui_components['config'] = config
            self.after_save_success(ui_components, config)
            [try_operation_safe(lambda cb=cb: cb(config)) for cb in self.callbacks]
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Unexpected save error: {str(e)}")
            self.after_save_failure(ui_components, f"Unexpected error: {str(e)}")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Reset config dengan comprehensive error handling dan generator cleanup"""
        try:
            self.logger.info(f"🔄 Starting reset config for {self.module_name}...")
            
            # Force close generators first
            generator_count = self._force_close_generators(ui_components)
            
            # Before reset hook
            self.before_reset(ui_components)
            
            # Get default config dengan error handling
            try:
                default_config = self.get_default_config()
                self.logger.debug(f"📋 Got default config with {len(default_config)} keys")
            except Exception as e:
                self.logger.error(f"💥 Default config failed: {str(e)}")
                self.after_reset_failure(ui_components, f"Default config error: {str(e)}")
                return False
            
            # Update UI dengan error handling
            try:
                self.update_ui(ui_components, default_config)
                self.logger.debug(f"🔄 UI updated with default config")
            except Exception as e:
                self.logger.error(f"💥 UI update failed: {str(e)}")
                self.after_reset_failure(ui_components, f"UI update error: {str(e)}")
                return False
            
            # Save default config dengan error handling
            try:
                success = self.config_manager.save_config(default_config, config_name or self.module_name)
                if not success:
                    raise Exception("Config manager returned False")
                    
                self.logger.success(f"✅ Config reset successfully")
            except Exception as e:
                self.logger.error(f"💥 Reset save failed: {str(e)}")
                self.after_reset_failure(ui_components, f"Reset save error: {str(e)}")
                return False
            
            # Success handling
            self._current_config = default_config
            ui_components['config'] = default_config
            self.after_reset_success(ui_components, default_config)
            [try_operation_safe(lambda cb=cb: cb(default_config)) for cb in self.callbacks]
            
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Unexpected reset error: {str(e)}")
            self.after_reset_failure(ui_components, f"Unexpected error: {str(e)}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk mendapatkan current config"""
        return self._current_config.copy()
    
    # Lifecycle hooks dengan enhanced logging
    def before_save(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum save dengan logging"""
        self.logger.debug("🔧 Preparing for save...")
        self._clear_ui_outputs(ui_components)
        self._update_status_panel(ui_components, "💾 Menyimpan konfigurasi...", "info")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah save berhasil dengan enhanced logging"""
        success_msg = "✅ Konfigurasi berhasil disimpan"
        self._update_status_panel(ui_components, success_msg, "success")
        self.logger.success(f"💾 Save completed for {self.module_name}")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah save gagal dengan enhanced logging"""
        error_msg = f"❌ Gagal menyimpan: {error}"
        self._update_status_panel(ui_components, error_msg, "error")
        self.logger.error(f"💥 Save failed for {self.module_name}: {error}")
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum reset dengan logging"""
        self.logger.debug("🔧 Preparing for reset...")
        self._clear_ui_outputs(ui_components)
        self._reset_progress_bars(ui_components)
        self._update_status_panel(ui_components, "🔄 Mereset konfigurasi...", "info")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah reset berhasil dengan enhanced logging"""
        success_msg = "✅ Konfigurasi berhasil direset"
        self._update_status_panel(ui_components, success_msg, "success")
        self.logger.success(f"🔄 Reset completed for {self.module_name}")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah reset gagal dengan enhanced logging"""
        error_msg = f"❌ Gagal reset: {error}"
        self._update_status_panel(ui_components, error_msg, "error")
        self.logger.error(f"💥 Reset failed for {self.module_name}: {error}")
    
    # Helper methods dengan enhanced error handling
    def _clear_ui_outputs(self, ui_components: Dict[str, Any]) -> None:
        """Clear UI outputs dengan error handling"""
        try:
            [widget.clear_output(wait=True) for key in ['log_output', 'status', 'confirmation_area'] 
             if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
        except Exception as e:
            self.logger.warning(f"⚠️ Clear outputs error: {str(e)}")
    
    def _reset_progress_bars(self, ui_components: Dict[str, Any]) -> None:
        """Reset progress bars dengan error handling"""
        try:
            # Hide progress bars
            [setattr(ui_components.get(key, type('', (), {'layout': type('', (), {'visibility': ''})()})()), 'layout.visibility', 'hidden')
             for key in ['progress_bar', 'progress_container', 'current_progress']]
            
            # Reset values
            [setattr(ui_components.get(key, type('', (), {'value': 0})()), 'value', 0)
             for key in ['progress_bar', 'current_progress']]
        except Exception as e:
            self.logger.warning(f"⚠️ Progress reset error: {str(e)}")
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
        """Update status panel dengan error handling"""
        try:
            show_status_safe(ui_components, message, status_type)
        except Exception as e:
            self.logger.warning(f"⚠️ Status update error: {str(e)}")
    
    # Callback management
    add_callback = lambda self, cb: self.callbacks.append(cb) if cb not in self.callbacks else None
    remove_callback = lambda self, cb: self.callbacks.remove(cb) if cb in self.callbacks else None


class BaseConfigHandler(ConfigHandler):
    """Base implementation dengan enhanced error handling"""
    
    def __init__(self, module_name: str, extract_fn: Optional[Callable] = None, 
                 update_fn: Optional[Callable] = None, parent_module: Optional[str] = None):
        super().__init__(module_name, parent_module)
        self.extract_fn = extract_fn
        self.update_fn = update_fn
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan enhanced error handling"""
        if self.extract_fn:
            try:
                return self.extract_fn(ui_components)
            except Exception as e:
                self.logger.error(f"💥 Extract function error: {str(e)}")
                raise
        
        # Fallback: auto-extract dari widgets
        config = {}
        try:
            for key, widget in ui_components.items():
                if hasattr(widget, 'value') and not key.startswith('_'):
                    config[key] = widget.value
        except Exception as e:
            self.logger.warning(f"⚠️ Auto-extract error: {str(e)}")
        
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan enhanced error handling"""
        if self.update_fn:
            try:
                return self.update_fn(ui_components, config)
            except Exception as e:
                self.logger.error(f"💥 Update function error: {str(e)}")
                raise
        
        # Fallback: auto-update widgets
        try:
            [setattr(widget, 'value', config[key]) 
             for key, widget in ui_components.items() 
             if hasattr(widget, 'value') and key in config]
        except Exception as e:
            self.logger.warning(f"⚠️ Auto-update error: {str(e)}")


# Factory functions
def create_config_handler(module_name: str, extract_fn: Callable = None, update_fn: Callable = None, 
                         parent_module: str = None) -> BaseConfigHandler:
    """Factory untuk BaseConfigHandler"""
    return BaseConfigHandler(module_name, extract_fn, update_fn, parent_module)

def get_or_create_handler(ui_components: Dict[str, Any], module_name: str, 
                         parent_module: str = None) -> ConfigHandler:
    """Get existing handler atau create default"""
    return (ui_components.get('config_handler') or 
            ui_components.setdefault('config_handler', 
                                   create_config_handler(module_name, parent_module=parent_module)))