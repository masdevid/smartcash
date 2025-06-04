"""
File: smartcash/ui/handlers/config_handlers.py
Deskripsi: Complete ConfigHandler dengan semua methods yang diperlukan termasuk load_config
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe

class ConfigHandler(ABC):
    """Enhanced ConfigHandler dengan complete lifecycle management dan load_config support."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.logger = get_logger(f"smartcash.ui.{self.full_module_name}.config")
        self.config_manager = get_config_manager()
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
            # Coba load dari defaults.py di folder handler module
            if hasattr(self, 'parent_module') and self.parent_module:
                module_path = f"smartcash.ui.{self.parent_module}.{self.module_name}.handlers.defaults"
            else:
                module_path = f"smartcash.ui.{self.module_name}.handlers.defaults"
            
            module = __import__(module_path, fromlist=['DEFAULT_CONFIG'])
            default_config = getattr(module, 'DEFAULT_CONFIG', {})
            
            if default_config:
                self.logger.info(f"📋 Loaded defaults.py for {self.module_name}")
                return default_config
            
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"🔍 defaults.py not found for {self.module_name}: {str(e)}")
        
        # Fallback: minimal default structure
        return {
            'module_name': self.module_name,
            'version': '1.0.0',
            'created_by': 'SmartCash',
            'settings': {}
        }
    
    def load_config(self, config_name: Optional[str] = None, use_base_config: bool = True) -> Dict[str, Any]:
        """Load config dengan base_config.yaml fallback dan inheritance resolution."""
        config_name = config_name or f"{self.module_name}_config"
        
        try:
            # 1. Coba load config spesifik terlebih dahulu
            specific_config = self.config_manager.get_config(config_name)
            
            if specific_config:
                # Jika ada config spesifik, resolve inheritance-nya
                resolved_config = self._resolve_config_inheritance(specific_config, config_name)
                self.logger.info(f"📄 Loaded specific config: {config_name}")
                return resolved_config
            
            # 2. Fallback ke base_config.yaml jika tidak ada config spesifik
            if use_base_config:
                base_config = self.config_manager.get_config('base_config')
                if base_config:
                    resolved_base = self._resolve_config_inheritance(base_config, 'base_config')
                    self.logger.info(f"📄 Loaded base_config.yaml for {config_name}")
                    return resolved_base
            
            # 3. Final fallback ke defaults.py untuk reset scenarios
            default_config = self.get_default_config()
            self.logger.warning(f"⚠️ Using defaults.py for {config_name}")
            return default_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Save config dengan lifecycle hooks"""
        try:
            # Before save hook
            self.before_save(ui_components)
            
            # Extract dan save config
            config = self.extract_config(ui_components)
            success = self.config_manager.save_config(config, config_name or f"{self.module_name}_config")
            
            if success:
                ui_components['config'] = config
                self.after_save_success(ui_components, config)
                [try_operation_safe(lambda cb=cb: cb(config)) for cb in self.callbacks]
            else:
                self.after_save_failure(ui_components, "Gagal menyimpan konfigurasi")
            
            return success
            
        except Exception as e:
            self.after_save_failure(ui_components, str(e))
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_name: Optional[str] = None) -> bool:
        """Reset config dengan lifecycle hooks"""
        try:
            # Before reset hook
            self.before_reset(ui_components)
            
            # Get default dan reset UI
            default_config = self.get_default_config()
            self.update_ui(ui_components, default_config)
            
            # Save default config
            success = self.config_manager.save_config(default_config, config_name or f"{self.module_name}_config")
            
            if success:
                ui_components['config'] = default_config
                self.after_reset_success(ui_components, default_config)
                [try_operation_safe(lambda cb=cb: cb(default_config)) for cb in self.callbacks]
            else:
                self.after_reset_failure(ui_components, "Gagal menyimpan default config")
            
            return success
            
        except Exception as e:
            self.after_reset_failure(ui_components, str(e))
            return False
    
    def _resolve_config_inheritance(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Resolve config inheritance dengan _base_ support"""
        if not config or '_base_' not in config:
            return config
        
        try:
            base_configs = config.pop('_base_')
            base_configs = [base_configs] if isinstance(base_configs, str) else base_configs
            
            # Merge semua base configs menggunakan ConfigManager
            merged_config = {}
            for base_name in base_configs:
                base_config = self.config_manager.get_config(base_name)
                if base_config:
                    # Recursive resolution untuk nested inheritance
                    resolved_base = self._resolve_config_inheritance(base_config, base_name)
                    merged_config.update(resolved_base)
                else:
                    self.logger.warning(f"⚠️ Base config tidak ditemukan: {base_name}")
            
            # Apply current config over merged base configs
            merged_config.update(config)
            self.logger.info(f"🔗 Resolved inheritance for {config_name}: {len(base_configs)} base configs")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"❌ Error resolving inheritance for {config_name}: {str(e)}")
            return config
    
    # Lifecycle hooks dengan default implementation
    def before_save(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum save - default: clear outputs dan reset UI state"""
        self._clear_ui_outputs(ui_components)
        self._update_status_panel(ui_components, "💾 Menyimpan konfigurasi...", "info")
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah save berhasil - default: update status dan log"""
        self._update_status_panel(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
        self.logger.success(f"💾 Konfigurasi {self.module_name} berhasil disimpan")
    
    def after_save_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah save gagal - default: update status dan log error"""
        self._update_status_panel(ui_components, f"❌ Gagal menyimpan: {error}", "error")
        self.logger.error(f"💥 Error saving config: {error}")
    
    def before_reset(self, ui_components: Dict[str, Any]) -> None:
        """Hook sebelum reset - default: clear outputs dan reset UI state"""
        self._clear_ui_outputs(ui_components)
        self._reset_progress_bars(ui_components)
        self._update_status_panel(ui_components, "🔄 Mereset konfigurasi...", "info")
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Hook setelah reset berhasil - default: update status dan log"""
        self._update_status_panel(ui_components, "✅ Konfigurasi berhasil direset", "success")
        self.logger.success(f"🔄 Konfigurasi {self.module_name} berhasil direset")
    
    def after_reset_failure(self, ui_components: Dict[str, Any], error: str) -> None:
        """Hook setelah reset gagal - default: update status dan log error"""
        self._update_status_panel(ui_components, f"❌ Gagal reset: {error}", "error")
        self.logger.error(f"💥 Error resetting config: {error}")
    
    # Helper methods dengan one-liner style
    def _clear_ui_outputs(self, ui_components: Dict[str, Any]) -> None:
        """Clear UI outputs."""
        [widget.clear_output(wait=True) for key in ['log_output', 'status', 'confirmation_area'] 
         if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    
    def _reset_progress_bars(self, ui_components: Dict[str, Any]) -> None:
        """Reset progress bars."""
        [setattr(ui_components.get(key, type('', (), {'layout': type('', (), {'visibility': ''})()})()), 'layout.visibility', 'hidden')
         for key in ['progress_bar', 'progress_container', 'current_progress']]
        [setattr(ui_components.get(key, type('', (), {'value': 0})()), 'value', 0)
         for key in ['progress_bar', 'current_progress']]
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
        """Update status panel dengan fallback."""
        show_status_safe(ui_components, message, status_type)
    
    # Callback management
    def add_callback(self, cb: Callable) -> None:
        """Add callback jika belum ada."""
        if cb not in self.callbacks:
            self.callbacks.append(cb)
    
    def remove_callback(self, cb: Callable) -> None:
        """Remove callback jika ada."""
        if cb in self.callbacks:
            self.callbacks.remove(cb)
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get config summary untuk display."""
        return f"📊 {self.module_name}: {len(config)} konfigurasi dimuat"
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config - override untuk custom validation"""
        return {'valid': True, 'errors': []}


class BaseConfigHandler(ConfigHandler):
    """Base implementation dengan parent module support dan extract/update yang fleksibel"""
    
    def __init__(self, module_name: str, extract_fn: Optional[Callable] = None, 
                 update_fn: Optional[Callable] = None, parent_module: Optional[str] = None):
        super().__init__(module_name, parent_module)
        self.extract_fn = extract_fn
        self.update_fn = update_fn
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan fallback ke function atau method"""
        if self.extract_fn:
            return self.extract_fn(ui_components)
        
        # Fallback: auto-extract dari widgets dengan 'value' attribute
        config = {}
        for key, widget in ui_components.items():
            if hasattr(widget, 'value') and not key.startswith('_'):
                config[key] = widget.value
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan fallback ke function atau method"""
        if self.update_fn:
            return self.update_fn(ui_components, config)
        
        # Fallback: auto-update widgets dengan 'value' attribute
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
        if not self.config_mapping:
            return super().extract_config(ui_components)
        
        return {config_key: getattr(ui_components.get(widget_key), 'value', None)
                for config_key, widget_key in self.config_mapping.items()
                if widget_key in ui_components}
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update dengan mapping atau auto-detection"""
        if not self.config_mapping:
            return super().update_ui(ui_components, config)
        
        [setattr(ui_components[widget_key], 'value', config.get(config_key))
         for config_key, widget_key in self.config_mapping.items()
         if widget_key in ui_components and config_key in config]


# Factory functions
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