"""
File: smartcash/ui/initializers/config_cell_initializer.py
Deskripsi: Clean ConfigCellInitializer dengan ConfigHandler integration dan parent module support
"""

from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import create_error_ui, show_status_safe
from smartcash.ui.handlers.config_handlers import ConfigHandler, BaseConfigHandler

class ConfigCellInitializer(ABC):
    """Generic ConfigCellInitializer dengan parent module support dan flexible callback system"""
    
    def __init__(self, module_name: str, config_filename: str, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        self.config_filename = config_filename
        self.logger = get_logger(f"smartcash.ui.{self.full_module_name}")
        self.parent_callbacks = {}  # Callbacks for different parent modules
        self.config_manager = get_config_manager()
        
        # Create config handler
        self.config_handler = self._create_config_handler(config_handler_class)
    
    def _create_config_handler(self, config_handler_class: Optional[Type[ConfigHandler]]) -> ConfigHandler:
        """Create config handler dengan parent module support"""
        if config_handler_class:
            # Check if constructor supports parent_module parameter
            try:
                return config_handler_class(self.module_name, self.parent_module)
            except TypeError:
                # Fallback for handlers that don't support parent_module yet
                return config_handler_class(self.module_name)
        
        # Fallback: create dengan extract/update methods dari subclass
        extract_fn = getattr(self, '_extract_config', None) if hasattr(self, '_extract_config') else None
        update_fn = getattr(self, '_update_ui', None) if hasattr(self, '_update_ui') else None
        
        return BaseConfigHandler(self.module_name, extract_fn, update_fn, self.parent_module)
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan ConfigHandler integration"""
        try:
            suppress_all_outputs()
            
            # Load config menggunakan ConfigHandler
            config = config or self.config_handler.load_config(self.config_filename)
            
            # Create UI components
            ui_components = self._create_config_ui(config, env, **kwargs)
            
            if not self._validate_ui(ui_components):
                return create_error_ui("Required components missing", self.module_name)
            
            # Add config handler dan setup handlers
            ui_components['config_handler'] = self.config_handler
            self._setup_handlers_with_config_handler(ui_components, config)
            
            show_status_safe(ui_components, f"✅ {self.module_name} ready", "success")
            return ui_components.get('main_container', ui_components)
            
        except Exception as e:
            return create_error_ui(f"Error: {str(e)}", self.module_name)
    
    def _setup_handlers_with_config_handler(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers menggunakan ConfigHandler dengan generic callback integration"""
        ui_components.update({'module_name': self.module_name, 'config': config, 'parent_module': self.parent_module})
        
        # Setup button handlers dengan ConfigHandler
        button_handlers = {
            'save_button': lambda b: self._save_config_with_callbacks(ui_components, b),
            'reset_button': lambda b: self._reset_config_with_callbacks(ui_components, b)
        }
        
        # Bind handlers dengan one-liner
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
        
        # Custom handlers hook
        getattr(self, '_setup_custom_handlers', lambda ui, cfg: None)(ui_components, config)
    
    def _save_config_with_callbacks(self, ui_components: Dict[str, Any], button) -> None:
        """Save config menggunakan ConfigHandler dengan generic callback integration"""
        button.disabled = True
        
        try:
            config_handler = ui_components['config_handler']
            success = config_handler.save_config(ui_components, self.config_filename)
            
            if success:
                config = ui_components.get('config', {})
                self._trigger_all_callbacks(config, 'save')
                
        except Exception as e:
            show_status_safe(ui_components, f"❌ Save error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _reset_config_with_callbacks(self, ui_components: Dict[str, Any], button) -> None:
        """Reset config menggunakan ConfigHandler dengan generic callback integration"""
        button.disabled = True
        
        try:
            config_handler = ui_components['config_handler']
            success = config_handler.reset_config(ui_components, self.config_filename)
            
            if success:
                config = ui_components.get('config', {})
                self._trigger_all_callbacks(config, 'reset')
                
        except Exception as e:
            show_status_safe(ui_components, f"❌ Reset error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _trigger_all_callbacks(self, config: Dict[str, Any], operation: str) -> None:
        """Trigger all registered callbacks dengan operation context"""
        # Parent-specific callbacks
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
    
    # Optional methods yang bisa di-override jika tidak menggunakan ConfigHandler
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
    
    # Generic callback management dengan parent module support
    def add_parent_callback(self, parent_type: str, callback: Callable) -> None:
        """Add callback untuk parent module tertentu (e.g., 'training', 'evaluation', 'inference')"""
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


# Generic ConfigHandler untuk config cells dengan parent module support
class GenericConfigCellHandler(ConfigHandler):
    """Generic ConfigHandler untuk config cells dengan flexible callback system"""
    
    def __init__(self, module_name: str, parent_callbacks: Optional[Dict[str, list]] = None, parent_module: Optional[str] = None):
        super().__init__(module_name, parent_module)
        self.parent_callbacks = parent_callbacks or {}
    
    def after_save_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan generic parent callback integration"""
        super().after_save_success(ui_components, config)
        self._trigger_parent_callbacks(config, 'save')
    
    def after_reset_success(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Override dengan generic parent callback integration"""
        super().after_reset_success(ui_components, config)
        self._trigger_parent_callbacks(config, 'reset')
    
    def _trigger_parent_callbacks(self, config: Dict[str, Any], operation: str) -> None:
        """Trigger parent callbacks dengan error handling"""
        for parent_type, callbacks in self.parent_callbacks.items():
            for cb in callbacks:
                try:
                    cb(config, operation) if len(cb.__code__.co_varnames) > 1 else cb(config)
                    self.logger.info(f"🔄 {parent_type} callback executed for {operation}")
                except Exception as e:
                    self.logger.warning(f"⚠️ {parent_type} callback error: {str(e)}")
    
    def add_parent_callback(self, parent_type: str, callback: Callable) -> None:
        """Add parent callback dengan one-liner"""
        self.parent_callbacks.setdefault(parent_type, []).append(callback) if callback not in self.parent_callbacks.get(parent_type, []) else None


# Factory functions dengan enhanced parent module support
def create_config_cell(initializer_class, module_name: str, config_filename: str, 
                      env=None, config=None, parent_module: Optional[str] = None,
                      parent_callbacks: Optional[Dict[str, Callable]] = None,
                      config_handler_class: Optional[Type[ConfigHandler]] = None, **kwargs) -> Any:
    """Enhanced factory dengan parent module dan callback support"""
    try:
        # Use GenericConfigCellHandler jika ada parent callbacks
        if parent_callbacks and not config_handler_class:
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
        return create_error_ui(f"Factory error: {str(e)}", module_name)


def connect_config_to_parent(config_initializer, parent_type: str, parent_ui_components: Dict[str, Any]) -> None:
    """Connect config cell ke parent UI dengan generic approach"""
    def parent_update_callback(new_config: Dict[str, Any], operation: str = 'update'):
        """Generic callback untuk parent UI updates"""
        try:
            # Generic config display update
            if f'{parent_type}_config_display' in parent_ui_components:
                display_updater = parent_ui_components.get(f'update_{parent_type}_info')
                if callable(display_updater):
                    display_updater(parent_ui_components, new_config)
            
            # Generic info display update
            if 'info_display' in parent_ui_components:
                info_updater = parent_ui_components.get('update_info_display')
                if callable(info_updater):
                    info_updater(parent_ui_components['info_display'], new_config)
            
            # Generic trigger update
            if hasattr(parent_ui_components, 'trigger_config_update'):
                parent_ui_components.trigger_config_update(new_config)
            
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


# One-liner utilities dengan parent module support
create_connected_config = lambda init_class, module, config_file, parent_type, parent_ui, parent_module=None, config_handler=None, **kw: create_config_cell(init_class, module, config_file, parent_module=parent_module, parent_callbacks={parent_type: lambda cfg: connect_config_to_parent(None, parent_type, parent_ui)}, config_handler_class=config_handler, **kw)

def get_parent_callback(parent_type: str, parent_ui: Dict[str, Any]) -> Callable:
    """Get parent callback dengan enhanced error handling"""
    def callback(cfg: Dict[str, Any], operation: str = 'update'):
        try:
            parent_ui.get(f'{parent_type}_config_update_callback', lambda x: None)(cfg)
            parent_ui['config'] = cfg
            print(f"✅ {parent_type} config updated via callback")
        except Exception as e:
            print(f"⚠️ {parent_type} callback error: {str(e)}")
    return callback

# Enhanced factory dengan multiple parent support
def create_config_cell_with_parents(initializer_class, module_name: str, config_filename: str,
                                   parent_connections: Dict[str, Dict[str, Any]], 
                                   parent_module: Optional[str] = None, **kwargs) -> Any:
    """Factory dengan multiple parent integration"""
    parent_callbacks = {
        parent_type: get_parent_callback(parent_type, parent_ui)
        for parent_type, parent_ui in parent_connections.items()
    }
    
    return create_config_cell(initializer_class, module_name, config_filename,
                             parent_module=parent_module,
                             parent_callbacks=parent_callbacks, **kwargs)

# Backward compatibility alias
create_config_cell_with_training = lambda init_class, module, config_file, training_ui, **kw: create_config_cell_with_parents(init_class, module, config_file, {'training': training_ui}, **kw)