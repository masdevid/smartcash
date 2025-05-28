"""
File: smartcash/ui/utils/config_cell_initializer.py
Deskripsi: Fixed config cell initializer dengan one-liner style yang functional
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import create_error_ui, show_status_safe


class ConfigCellInitializer(ABC):
    """Fixed config cell initializer dengan one-liner style yang functional"""
    
    def __init__(self, module_name: str, config_filename: str):
        self.module_name = module_name
        self.config_filename = config_filename
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        self.config_callbacks = []
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan safe error handling"""
        try:
            suppress_all_outputs()
            config = self._load_config(config)
            ui_components = self._create_config_ui(config, env, **kwargs)
            
            if not self._validate_ui(ui_components):
                return create_error_ui("Required components missing", self.module_name)
            
            self._setup_handlers(ui_components, config)
            show_status_safe(f"✅ {self.module_name} ready", "success", ui_components)
            return ui_components.get('main_container', ui_components)
            
        except Exception as e:
            return create_error_ui(f"Error: {str(e)}", self.module_name)
    
    # Abstract methods
    @abstractmethod
    def _create_config_ui(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        pass
    
    # One-liner helper methods
    _load_config = lambda self, config: config or get_config_manager().get_config(self.config_filename) or self._get_default_config()
    _validate_ui = lambda self, ui_components: all(comp in ui_components for comp in ['save_button', 'reset_button', 'status_panel'])
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers dengan one-liner button registration"""
        ui_components.update({'module_name': self.module_name, 'config': config})
        
        # One-liner button handlers
        getattr(ui_components.get('save_button'), 'on_click', lambda x: None)(lambda b: self._save_config(ui_components, b))
        getattr(ui_components.get('reset_button'), 'on_click', lambda x: None)(lambda b: self._reset_config(ui_components, b))
        
        # Custom handlers hook
        getattr(self, '_setup_custom_handlers', lambda ui, cfg: None)(ui_components, config)
    
    def _save_config(self, ui_components: Dict[str, Any], button) -> None:
        """Save config dengan proper success/error handling"""
        button.disabled = True
        self._update_status_panel(ui_components, "💾 Saving...", "info")
        
        try:
            config = self._extract_config(ui_components)
            success = get_config_manager().save_config(config, self.config_filename)
            
            if success:
                ui_components['config'] = config
                self._update_status_panel(ui_components, "✅ Saved", "success")
                [cb(config) for cb in self.config_callbacks]
            else:
                self._update_status_panel(ui_components, "❌ Save failed", "error")
             
        except Exception as e:
            self._update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _reset_config(self, ui_components: Dict[str, Any], button) -> None:
        """Reset config dengan one-liner success handling"""
        button.disabled = True
        self._update_status_panel(ui_components, "🔄 Resetting...", "info")
        
        try:
            default_config = self._get_default_config()
            self._update_ui(ui_components, default_config)
            get_config_manager().save_config(default_config, self.config_filename)
            
            # One-liner success handling
            ui_components.update({'config': default_config})
            self._update_status_panel(ui_components, "✅ Reset complete", "success")
            [cb(default_config) for cb in self.config_callbacks]
            
        except Exception as e:
            self._update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
        """Update status panel dengan one-liner fallback"""
        try:
            from smartcash.ui.utils.fallback_utils import create_status_message
            
            status_panel = ui_components.get('status_panel')
            (setattr(status_panel, 'value', create_status_message(message, status_type=status_type, show_icon=True)) 
             if status_panel and hasattr(status_panel, 'value') else 
             show_status_safe(message, status_type, ui_components))
             
        except Exception:
            # One-liner fallback print
            print(f"{'✅' if status_type == 'success' else '⚠️' if status_type == 'warning' else '❌' if status_type == 'error' else 'ℹ️'} {message}")
    
    # One-liner callback management
    add_callback = lambda self, cb: self.config_callbacks.append(cb) if cb not in self.config_callbacks else None
    remove_callback = lambda self, cb: self.config_callbacks.remove(cb) if cb in self.config_callbacks else None


def create_config_cell(initializer_class, module_name: str, config_filename: str, 
                      env=None, config=None, **kwargs) -> Any:
    """Factory untuk create config cell dengan safe error handling"""
    try:
        return initializer_class(module_name, config_filename).initialize(env, config, **kwargs)
    except Exception as e:
        return create_error_ui(f"Factory error: {str(e)}", module_name)