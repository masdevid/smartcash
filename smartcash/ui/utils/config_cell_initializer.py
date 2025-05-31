"""
File: smartcash/ui/utils/config_cell_initializer.py
Deskripsi: Fixed config cell initializer dengan training callback integration
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import create_error_ui, show_status_safe


class ConfigCellInitializer(ABC):
    """Fixed config cell initializer dengan training integration callbacks"""
    
    def __init__(self, module_name: str, config_filename: str):
        self.module_name = module_name
        self.config_filename = config_filename
        self.logger = get_logger(f"smartcash.ui.{module_name}")
        self.config_callbacks = []
        self.training_ui_callback = None  # Callback untuk training UI updates
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan training integration"""
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
    
    def set_training_ui_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback untuk training UI updates"""
        self.training_ui_callback = callback
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers dengan training integration"""
        ui_components.update({'module_name': self.module_name, 'config': config})
        
        # One-liner button handlers dengan training callback
        getattr(ui_components.get('save_button'), 'on_click', lambda x: None)(lambda b: self._save_config_with_training_update(ui_components, b))
        getattr(ui_components.get('reset_button'), 'on_click', lambda x: None)(lambda b: self._reset_config_with_training_update(ui_components, b))
        
        # Custom handlers hook
        getattr(self, '_setup_custom_handlers', lambda ui, cfg: None)(ui_components, config)
    
    def _save_config_with_training_update(self, ui_components: Dict[str, Any], button) -> None:
        """Save config dengan training UI callback integration"""
        button.disabled = True
        self._update_status_panel(ui_components, "💾 Saving configuration...", "info")
        
        try:
            config = self._extract_config(ui_components)
            success = get_config_manager().save_config(config, self.config_filename)
            
            if success:
                ui_components['config'] = config
                self._update_status_panel(ui_components, "✅ Configuration saved", "success")
                
                # Trigger all callbacks
                [cb(config) for cb in self.config_callbacks]
                
                # Trigger training UI update jika callback tersedia
                if self.training_ui_callback:
                    try:
                        self.training_ui_callback(config)
                        self.logger.info("🔄 Training UI updated with new config")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Training callback error: {str(e)}")
            else:
                self._update_status_panel(ui_components, "❌ Save failed", "error")
             
        except Exception as e:
            self._update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    def _reset_config_with_training_update(self, ui_components: Dict[str, Any], button) -> None:
        """Reset config dengan training UI update"""
        button.disabled = True
        self._update_status_panel(ui_components, "🔄 Resetting configuration...", "info")
        
        try:
            default_config = self._get_default_config()
            self._update_ui(ui_components, default_config)
            get_config_manager().save_config(default_config, self.config_filename)
            
            # One-liner success handling dengan training callback
            ui_components.update({'config': default_config})
            self._update_status_panel(ui_components, "✅ Reset complete", "success")
            [cb(default_config) for cb in self.config_callbacks]
            
            # Trigger training UI update
            if self.training_ui_callback:
                try:
                    self.training_ui_callback(default_config)
                    self.logger.info("🔄 Training UI reset with default config")
                except Exception as e:
                    self.logger.warning(f"⚠️ Training reset callback error: {str(e)}")
            
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
    trigger_config_update = lambda self, config: [cb(config) for cb in self.config_callbacks] + ([self.training_ui_callback(config)] if self.training_ui_callback else [])


def create_config_cell(initializer_class, module_name: str, config_filename: str, 
                      env=None, config=None, training_ui_callback: Optional[Callable] = None, **kwargs) -> Any:
    """Factory untuk create config cell dengan training integration"""
    try:
        initializer = initializer_class(module_name, config_filename)
        
        # Set training callback jika disediakan
        if training_ui_callback:
            initializer.set_training_ui_callback(training_ui_callback)
        
        return initializer.initialize(env, config, **kwargs)
    except Exception as e:
        return create_error_ui(f"Factory error: {str(e)}", module_name)


def connect_config_to_training(config_initializer, training_ui_components: Dict[str, Any]) -> None:
    """Connect config cell ke training UI untuk automatic updates"""
    def training_update_callback(new_config: Dict[str, Any]):
        """Callback untuk update training UI dengan config baru"""
        try:
            # Update training config display
            if 'training_config_display' in training_ui_components:
                from smartcash.ui.training.utils.training_display_utils import update_training_info
                update_training_info(training_ui_components, new_config)
            
            # Update info display
            if 'info_display' in training_ui_components:
                from smartcash.ui.training.components.training_form import update_info_display
                update_info_display(training_ui_components['info_display'], new_config)
            
            # Trigger training initializer config update jika ada
            if hasattr(training_ui_components, 'trigger_config_update'):
                training_ui_components.trigger_config_update(new_config)
            
        except Exception as e:
            print(f"⚠️ Training config update error: {str(e)}")
    
    # Set callback ke config initializer
    if hasattr(config_initializer, 'set_training_ui_callback'):
        config_initializer.set_training_ui_callback(training_update_callback)


# One-liner utilities untuk quick integration
create_connected_config = lambda init_class, module, config_file, training_ui, **kw: create_config_cell(init_class, module, config_file, training_ui_callback=lambda cfg: connect_config_to_training(None, training_ui), **kw)
get_training_callback = lambda training_ui: lambda cfg: training_ui.get('config_update_callback', lambda x: None)(cfg)