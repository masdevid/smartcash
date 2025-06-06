"""
File: smartcash/ui/handlers/config_handlers.py
Deskripsi: Fixed ConfigHandler dengan proper log output access dan error handling
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe

class ConfigHandler(ABC):
    """Fixed ConfigHandler dengan proper log output handling dan complete lifecycle management."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name, self.parent_module = module_name, parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.logger, self.config_manager = get_logger(f"smartcash.ui.{self.full_module_name}.config"), get_config_manager()
        self.callbacks = []
        
    @abstractmethod
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dari komponen UI"""
        pass
        
    @abstractmethod
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari konfigurasi yang dimuat"""
        pass
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py untuk reset scenarios"""
        try:
            module_path = (f"smartcash.ui.{self.parent_module}.{self.module_name}.handlers.defaults" 
                          if hasattr(self, 'parent_module') and self.parent_module 
                          else f"smartcash.ui.{self.module_name}.handlers.defaults")
            
            module = __import__(module_path, fromlist=['DEFAULT_CONFIG'])
            default_config = getattr(module, 'DEFAULT_CONFIG', {})
            
            return (self.logger.info(f"📋 Loaded defaults.py for {self.module_name}") or default_config 
                   if default_config else self._get_fallback_config())
            
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"🔍 defaults.py not found for {self.module_name}: {str(e)}")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback minimal default structure"""
        return {'module_name': self.module_name, 'version': '1.0.0', 'created_by': 'SmartCash', 'settings': {}}
    
    def load_config(self, config_name: Optional[str] = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan base_config.yaml fallback dan inheritance resolution."""
        config_name = config_name or f"{self.module_name}_config"
        
        try:
            # One-liner untuk specific config dengan early return
            if specific_config := self.config_manager.get_config(config_name):
                self.logger.info(f"📄 Loaded specific config: {config_name}")
                return self._resolve_config_inheritance(specific_config, config_name)
            
            # One-liner untuk base config fallback
            if use_base_config and (base_config := self.config_manager.get_config('base_config')):
                self.logger.info(f"📄 Loaded base_config.yaml for {config_name}")
                return self._resolve_config_inheritance(base_config, 'base_config')
            
            # Final fallback
            return (self.logger.warning(f"⚠️ Using defaults.py for {config_name}") or 
                   self.get_default_config())
            
        except Exception as e:
            return (self.logger.warning(f"⚠️ Error loading config: {str(e)}") or 
                   self.get_default_config())
    
    def save_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Save config dengan lifecycle hooks dan fixed log output"""
        try:
            self.before_save(ui_components)
            config = self.extract_config(ui_components)
            success = self.config_manager.save_config(config, config_name or f"{self.module_name}_config")
            
            # One-liner untuk success/failure handling
            return (self._handle_save_success(ui_components, config) if success 
                   else self._handle_save_failure(ui_components, "Gagal menyimpan konfigurasi")) or success
            
        except Exception as e:
            return self._handle_save_failure(ui_components, str(e)) or False
    
    def reset_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Reset config dengan lifecycle hooks dan fixed log output"""
        try:
            self.before_reset(ui_components)
            default_config = self.get_default_config()
            self.update_ui(ui_components, default_config)
            success = self.config_manager.save_config(default_config, config_name or f"{self.module_name}_config")
            
            # One-liner untuk success/failure handling
            return (self._handle_reset_success(ui_components, default_config) if success 
                   else self._handle_reset_failure(ui_components, "Gagal menyimpan default config")) or success
            
        except Exception as e:
            return self._handle_reset_failure(ui_components, str(e)) or False
    
    def _handle_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Handle save success dengan callback execution"""
        ui_components['config'] = config
        self.after_save_success(ui_components, config)
        [try_operation_safe(lambda cb=cb: cb(config)) for cb in self.callbacks]
    
    def _handle_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Handle save failure"""
        self.after_save_failure(ui_components, error)
    
    def _handle_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Handle reset success dengan callback execution"""
        ui_components['config'] = config
        self.after_reset_success(ui_components, config)
        [try_operation_safe(lambda cb=cb: cb(config)) for cb in self.callbacks]
    
    def _handle_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Handle reset failure"""
        self.after_reset_failure(ui_components, error)
    
    def _resolve_config_inheritance(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Resolve config inheritance dengan _base_ support"""
        if not config or '_base_' not in config:
            return config
        
        try:
            base_configs = ([config.pop('_base_')] if isinstance(base_configs := config.pop('_base_'), str) 
                          else base_configs)
            
            # One-liner untuk merge base configs
            merged_config = {}
            [merged_config.update(self._resolve_config_inheritance(base_config, base_name)) 
             for base_name in base_configs 
             if (base_config := self.config_manager.get_config(base_name)) 
             or self.logger.warning(f"⚠️ Base config tidak ditemukan: {base_name}")]
            
            merged_config.update(config)
            self.logger.info(f"🔗 Resolved inheritance for {config_name}: {len(base_configs)} base configs")
            return merged_config
            
        except Exception as e:
            return (self.logger.error(f"❌ Error resolving inheritance for {config_name}: {str(e)}") or 
                   config)
    
    # Lifecycle hooks dengan default implementation
    def before_save(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum save"""
        self._clear_ui_outputs(ui_components)
        self._update_status_panel(ui_components, "💾 Menyimpan konfigurasi...", "info")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah save berhasil"""
        self._update_status_panel(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
        self.logger.success(f"💾 Konfigurasi {self.module_name} berhasil disimpan")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah save gagal"""
        self._update_status_panel(ui_components, f"❌ Gagal menyimpan: {error}", "error")
        self.logger.error(f"💥 Error saving config: {error}")
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum reset"""
        self._clear_ui_outputs(ui_components), self._reset_progress_bars(ui_components)
        self._update_status_panel(ui_components, "🔄 Mereset konfigurasi...", "info")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah reset berhasil"""
        self._update_status_panel(ui_components, "✅ Konfigurasi berhasil direset", "success")
        self.logger.success(f"🔄 Konfigurasi {self.module_name} berhasil direset")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah reset gagal"""
        self._update_status_panel(ui_components, f"❌ Gagal reset: {error}", "error")
        self.logger.error(f"💥 Error resetting config: {error}")
    
    # Fixed helper methods dengan proper log output access
    def _clear_ui_outputs(self, ui_components: Dict[str, Any]) -> None:
        """Clear UI outputs dengan safe widget access"""
        output_keys = ['log_output', 'status', 'confirmation_area']
        
        for key in output_keys:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'clear_output'):
                try:
                    widget.clear_output(wait=True)
                except Exception as e:
                    self.logger.debug(f"🔍 Error clearing {key}: {str(e)}")
    
    def _reset_progress_bars(self, ui_components: Dict[str, Any]) -> None:
        """Reset progress bars dengan safe widget access"""
        progress_keys = ['progress_bar', 'progress_container', 'current_progress', 'progress_tracker']
        
        for key in progress_keys:
            widget = ui_components.get(key)
            if widget:
                try:
                    # Try to hide widget
                    if hasattr(widget, 'layout'):
                        widget.layout.visibility = 'hidden'
                        widget.layout.display = 'none'
                    
                    # Try to reset value
                    if hasattr(widget, 'value'):
                        widget.value = 0
                    
                    # Try to reset progress tracker
                    if hasattr(widget, 'reset'):
                        widget.reset()
                        
                except Exception as e:
                    self.logger.debug(f"🔍 Error resetting {key}: {str(e)}")
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
        """Update status panel dengan fallback dan safe log output access"""
        # Primary: Use show_status_safe
        show_status_safe(message, status_type, ui_components)
        
        # Secondary: Try to update log_output safely
        try:
            log_output = ui_components.get('log_output')
            if log_output and hasattr(log_output, '__enter__'):
                with log_output:
                    from IPython.display import display, HTML
                    color_map = {'success': '#28a745', 'info': '#007bff', 'warning': '#ffc107', 'error': '#dc3545'}
                    color = color_map.get(status_type, '#007bff')
                    display(HTML(f"<div style='color: {color}; margin: 5px 0; padding: 8px; background: {color}10; border-radius: 4px;'>{message}</div>"))
        except Exception as e:
            self.logger.debug(f"🔍 Log output display error: {str(e)}")
    
    # Callback management dengan one-liner checks
    def add_callback(self, cb: Callable) -> None:
        """Add callback jika belum ada"""
        cb not in self.callbacks and self.callbacks.append(cb)
    
    def remove_callback(self, cb: Callable) -> None:
        """Remove callback jika ada"""
        cb in self.callbacks and self.callbacks.remove(cb)
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get config summary untuk display"""
        return f"📊 {self.module_name}: {len(config)} konfigurasi dimuat"
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config - override untuk custom validation"""
        return {'valid': True, 'errors': []}


class BaseConfigHandler(ConfigHandler):
    """Base implementation dengan parent module support dan extract/update yang fleksibel"""
    
    def __init__(self, module_name: str, extract_fn: Optional[Callable] = None, 
                 update_fn: Optional[Callable] = None, parent_module: Optional[str] = None):
        super().__init__(module_name, parent_module)
        self.extract_fn, self.update_fn = extract_fn, update_fn
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan fallback ke function atau auto-extract"""
        return (self.extract_fn(ui_components) if self.extract_fn 
               else {key: widget.value for key, widget in ui_components.items() 
                     if hasattr(widget, 'value') and not key.startswith('_')})
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan fallback ke function atau auto-update"""
        if self.update_fn:
            return self.update_fn(ui_components, config)
        
        # One-liner auto-update dengan walrus operator
        [setattr(widget, 'value', config[key]) 
         for key, widget in ui_components.items() 
         if hasattr(widget, 'value') and key in config]


class SimpleConfigHandler(BaseConfigHandler):
    """Simple config handler dengan parent module support untuk kasus sederhana"""
    
    def __init__(self, module_name: str, config_mapping: Optional[Dict[str, str]] = None, 
                 parent_module: Optional[str] = None):
        super().__init__(module_name, parent_module=parent_module)
        self.config_mapping = config_mapping or {}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dengan mapping atau auto-detection"""
        return (super().extract_config(ui_components) if not self.config_mapping
               else {config_key: getattr(ui_components.get(widget_key), 'value', None)
                     for config_key, widget_key in self.config_mapping.items()
                     if widget_key in ui_components})
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update dengan mapping atau auto-detection"""
        if not self.config_mapping:
            return super().update_ui(ui_components, config)
        
        # One-liner update dengan mapping
        [setattr(ui_components[widget_key], 'value', config.get(config_key))
         for config_key, widget_key in self.config_mapping.items()
         if widget_key in ui_components and config_key in config]


# Factory functions dengan one-liner returns
def create_config_handler(module_name: str, extract_fn: Callable = None, update_fn: Callable = None, 
                         parent_module: str = None) -> BaseConfigHandler:
    """Factory untuk BaseConfigHandler dengan parent module support"""
    return BaseConfigHandler(module_name, extract_fn, update_fn, parent_module)

def create_simple_handler(module_name: str, mapping: Dict[str, str] = None, 
                         parent_module: str = None) -> SimpleConfigHandler:
    """Factory untuk SimpleConfigHandler dengan parent module support"""
    return SimpleConfigHandler(module_name, mapping, parent_module)

def get_or_create_handler(ui_components: Dict[str, Any], module_name: str, 
                         parent_module: str = None) -> ConfigHandler:
    """Get existing handler atau create default dengan parent module support"""
    return (ui_components.get('config_handler') or 
            ui_components.setdefault('config_handler', 
                                   create_config_handler(module_name, parent_module=parent_module)))