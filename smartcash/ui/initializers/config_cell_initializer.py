"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Optimized ConfigCellInitializer dengan fixed save/reset status updates dan error handling
"""

from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
from unittest import result
import ipywidgets as widgets
import sys
from IPython.display import display
import traceback

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import create_error_ui, show_status_safe
from smartcash.ui.handlers.config_handlers import ConfigHandler, BaseConfigHandler

class ConfigCellInitializer(ABC):
    """Optimized ConfigCellInitializer dengan improved status updates dan error handling"""
    
    def __init__(self, module_name: str, config_filename: str, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.config_filename = config_filename
        self.logger = get_logger(f"smartcash.ui.{self.full_module_name}")
        self.parent_callbacks = {}
        self.config_manager = get_config_manager()
        
        # Create config handler
        self.config_handler = self._create_config_handler(config_handler_class)
    
    def _create_config_handler(self, config_handler_class: Optional[Type[ConfigHandler]]) -> ConfigHandler:
        """Create config handler dengan parent module support"""
        if config_handler_class:
            try:
                return config_handler_class(self.module_name, self.parent_module)
            except TypeError:
                return config_handler_class(self.module_name)
        
        # Fallback: create dengan extract/update methods dari subclass
        extract_fn = getattr(self, '_extract_config', None) if hasattr(self, '_extract_config') else None
        update_fn = getattr(self, '_update_ui', None) if hasattr(self, '_update_ui') else None
        
        return BaseConfigHandler(self.module_name, extract_fn, update_fn, self.parent_module)
    
    def _create_fallback_ui(self, error_msg: str, exc_info=None) -> Dict[str, Any]:
        """Membuat fallback UI dengan error handling yang sederhana
        
        Args:
            error_msg: Pesan error yang akan ditampilkan
            exc_info: Optional exception info tuple (type, value, traceback)
            
        Returns:
            Dictionary berisi komponen UI fallback
        """
        from smartcash.ui.utils.fallback_utils import create_fallback_ui, FallbackConfig
        
        return create_fallback_ui(
            error_message=error_msg,
            module_name=self.module_name,
            exc_info=exc_info,
            config=FallbackConfig(
                title=f"⚠️ Error in {self.module_name}",
                module_name=self.module_name,
                traceback=traceback.format_exc() if exc_info else ""
            )
        )
    
    def handle_ui_exception(self, error: Exception, context: str = "UI") -> Dict[str, Any]:
        """Menangani exception yang terjadi saat membuat UI
        
        Args:
            error: Exception yang terjadi
            context: Konteks error (default: "UI")
            
        Returns:
            Dictionary berisi komponen UI fallback
        """
        error_msg = f"Gagal membuat {context} {self.module_name}: {str(error)}"
        self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return self._create_fallback_ui(
            error_msg=error_msg,
            exc_info=sys.exc_info()
        )
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Optimized initialization dengan proper error handling"""
        try:
            self.logger.debug(f"Memulai inisialisasi {self.module_name}...")
            suppress_all_outputs()
            
            # Load config menggunakan ConfigHandler
            self.logger.debug(f"Mencoba load config untuk {self.config_filename}...")
            if config is None:
                self.logger.debug("Config tidak disediakan, memuat dari config handler...")
                config = self.config_handler.load_config(self.config_filename)
                self.logger.debug(f"Config berhasil dimuat: {config is not None}")
            else:
                self.logger.debug("Menggunakan config yang disediakan")
            
            # Create UI components
            self.logger.debug("Membuat komponen UI...")
            ui_components = self._create_config_ui(config, env, **kwargs)
            
            if not self._validate_ui(ui_components):
                self.logger.warning("Validasi UI gagal, membuat fallback UI...")
                fallback = self._create_fallback_ui("Komponen yang diperlukan tidak ditemukan")
                return fallback.get('ui', fallback)
            
            # Add config handler dan setup handlers
            self.logger.debug("Menyiapkan handler...")
            ui_components['config_handler'] = self.config_handler
            self._setup_handlers_with_config_handler(ui_components, config)
            
            show_status_safe(f"{self.module_name.capitalize()} siap digunakan", "success", ui_components)
            
            # Kembalikan main_container jika ada, jika tidak kembalikan ui_components
            result = ui_components.get('main_container', ui_components)
            self.logger.debug("Inisialisasi selesai")
            return result
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi {self.module_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            fallback = self._create_fallback_ui(error_msg, exc_info=sys.exc_info())
            return fallback.get('ui', fallback)
    
    def _setup_handlers_with_config_handler(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers dengan optimized button state management"""
        ui_components.update({'module_name': self.module_name, 'config': config, 'parent_module': self.parent_module})
        
        # Setup button handlers dengan improved error handling
        button_handlers = {
            'save_button': lambda b: self._save_config_with_status(ui_components, b),
            'reset_button': lambda b: self._reset_config_with_status(ui_components, b)
        }
        
        # Bind handlers dengan one-liner
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
        
        # Custom handlers hook
        getattr(self, '_setup_custom_handlers', lambda ui, cfg: None)(ui_components, config)
    
    def _save_config_with_status(self, ui_components: Dict[str, Any], button) -> None:
        """Save config dengan improved status updates dan error handling"""
        button.disabled = True
        
        try:
            show_status_safe("💾 Menyimpan konfigurasi...", "info", ui_components)
            
            config_handler = ui_components['config_handler']
            success = config_handler.save_config(ui_components, self.config_filename)
            
            if success:
                config = ui_components.get('config', {})
                show_status_safe("✅ Konfigurasi berhasil disimpan", "success", ui_components)
                self._trigger_all_callbacks(config, 'save')
                self.logger.success(f"💾 {self.module_name} config tersimpan")
            else:
                show_status_safe("❌ Gagal menyimpan konfigurasi", "error", ui_components)
                self.logger.error(f"💥 Gagal save {self.module_name} config")
                
        except Exception as e:
            error_msg = f"❌ Error saat menyimpan: {str(e)}"
            show_status_safe(error_msg, "error", ui_components)
            self.logger.error(f"💥 Save error {self.module_name}: {str(e)}")
        finally:
            button.disabled = False
    
    def _reset_config_with_status(self, ui_components: Dict[str, Any], button) -> None:
        """Reset config dengan improved status updates dan error handling"""
        button.disabled = True
        
        try:
            show_status_safe("🔄 Mereset konfigurasi...", "info", ui_components)
            
            config_handler = ui_components['config_handler']
            success = config_handler.reset_config(ui_components, self.config_filename)
            
            if success:
                config = ui_components.get('config', {})
                show_status_safe("✅ Konfigurasi berhasil direset", "success", ui_components)
                self._trigger_all_callbacks(config, 'reset')
                self.logger.success(f"🔄 {self.module_name} config direset")
            else:
                show_status_safe("❌ Gagal reset konfigurasi", "error", ui_components)
                self.logger.error(f"💥 Gagal reset {self.module_name} config")
                
        except Exception as e:
            error_msg = f"❌ Error saat reset: {str(e)}"
            show_status_safe(error_msg, "error", ui_components)
            self.logger.error(f"💥 Reset error {self.module_name}: {str(e)}")
        finally:
            button.disabled = False
    
    def _trigger_all_callbacks(self, config: Dict[str, Any], operation: str) -> None:
        """Trigger all registered callbacks dengan operation context"""
        for parent_type, callbacks in self.parent_callbacks.items():
            for cb in callbacks:
                try:
                    cb(config, operation) if callable(cb) and len(cb.__code__.co_varnames) > 1 else cb(config)
                    self.logger.info(f"🔄 {parent_type} callback executed for {operation}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {parent_type} callback error: {str(e)}")
    
    # Abstract methods
    @abstractmethod
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components untuk config"""
        pass
    
    # Optional methods
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI - fallback jika ConfigHandler tidak handle"""
        raise NotImplementedError("_extract_config harus diimplementasikan jika ConfigHandler tidak menangani extract")
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config - fallback jika ConfigHandler tidak handle"""
        raise NotImplementedError("_update_ui harus diimplementasikan jika ConfigHandler tidak menangani update")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config - fallback jika ConfigHandler tidak handle"""
        return self.config_handler.get_default_config()
    
    # Helper methods dengan one-liner style
    _validate_ui = lambda self, ui_components: all(comp in ui_components for comp in ['save_button', 'reset_button'])
    
    # Callback management dengan parent module support
    def add_parent_callback(self, parent_type: str, callback: Callable) -> None:
        """Add callback untuk parent module tertentu"""
        if parent_type not in self.parent_callbacks:
            self.parent_callbacks[parent_type] = []
        
        if callback not in self.parent_callbacks[parent_type]:
            self.parent_callbacks[parent_type].append(callback)
            self.logger.info(f"📝 Added {parent_type} callback for {self.full_module_name}")
    
    def remove_parent_callback(self, parent_type: str, callback: Callable) -> None:
        """Remove callback dari parent module tertentu"""
        if parent_type in self.parent_callbacks and callback in self.parent_callbacks[parent_type]:
            self.parent_callbacks[parent_type].remove(callback)
            self.logger.info(f"🗑️ Removed {parent_type} callback for {self.full_module_name}")
    
    def set_parent_ui_callback(self, parent_type: str, callback: Callable) -> None:
        """Set callback untuk parent UI updates (backward compatibility)"""
        self.add_parent_callback(parent_type, callback)
    
    def trigger_config_update(self, config: Dict[str, Any], operation: str = 'update') -> None:
        """Trigger all callbacks dengan operation context"""
        self._trigger_all_callbacks(config, operation)
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information dengan parent context"""
        return {
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'full_module_name': self.full_module_name,
            'config_filename': self.config_filename,
            'parent_callbacks': {k: len(v) for k, v in self.parent_callbacks.items()}
        }


# Factory functions remain the same dengan enhanced error handling
def create_config_cell(initializer_class, module_name: str, config_filename: str, 
                      env=None, config=None, parent_module: Optional[str] = None,
                      parent_callbacks: Optional[Dict[str, Callable]] = None,
                      config_handler_class: Optional[Type[ConfigHandler]] = None, **kwargs) -> Any:
    """Enhanced factory dengan improved error handling"""
    try:
        # Use GenericConfigCellHandler jika ada parent callbacks
        if parent_callbacks and not config_handler_class:
            from .config_cell_initializer import GenericConfigCellHandler
            parent_callback_dict = {k: [v] if callable(v) else v for k, v in parent_callbacks.items()}
            config_handler_class = lambda mn, pm=None: GenericConfigCellHandler(mn, parent_callback_dict, pm)
        
        initializer = initializer_class(module_name, config_filename, config_handler_class, parent_module)
        
        # Set parent callbacks
        if parent_callbacks:
            for parent_type, callback in parent_callbacks.items():
                if callable(callback):
                    initializer.add_parent_callback(parent_type, callback)
                elif isinstance(callback, list):
                    [initializer.add_parent_callback(parent_type, cb) for cb in callback]
        
        return initializer.initialize(env, config, **kwargs)
        
    except Exception as e:
        logger = get_logger(f"smartcash.ui.{module_name}")
        logger.error(f"❌ Factory error {module_name}: {str(e)}")
        return create_error_ui(f"Factory error: {str(e)}", module_name)


# Additional utility functions untuk backward compatibility
def connect_config_to_parent(config_initializer, parent_type: str, parent_ui_components: Dict[str, Any]) -> None:
    """Connect config cell ke parent UI dengan generic approach"""
    def parent_update_callback(new_config: Dict[str, Any], operation: str = 'update'):
        try:
            # Generic config display update
            if f'{parent_type}_config_display' in parent_ui_components:
                display_updater = parent_ui_components.get(f'update_{parent_type}_info')
                if callable(display_updater):
                    display_updater(parent_ui_components, new_config)
            
            # Update config di parent components
            parent_ui_components['config'] = new_config
            print(f"✅ {parent_type} UI updated with {operation} operation")
            
        except Exception as e:
            print(f"⚠️ {parent_type} config update error: {str(e)}")
    
    # Set callback dengan generic approach
    if hasattr(config_initializer, 'add_parent_callback'):
        config_initializer.add_parent_callback(parent_type, parent_update_callback)
    elif hasattr(config_initializer, 'set_parent_ui_callback'):
        config_initializer.set_parent_ui_callback(parent_type, parent_update_callback)