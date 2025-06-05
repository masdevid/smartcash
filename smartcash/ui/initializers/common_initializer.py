"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Fixed CommonInitializer dengan proper handler binding dan tanpa generator error
"""

from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
import datetime

from smartcash.ui.utils.fallback_utils import create_fallback_ui, try_operation_safe, show_status_safe
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge, get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.utils.ui_logger_namespace import KNOWN_NAMESPACES, register_namespace
from smartcash.ui.handlers.config_handlers import ConfigHandler

class CommonInitializer(ABC):
    """Fixed CommonInitializer dengan proper button handler binding"""
    
    def __init__(self, module_name: str, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Setup logger namespace
        self.logger_namespace = (KNOWN_NAMESPACES.get(f"smartcash.ui.{self.full_module_name}") or 
                                KNOWN_NAMESPACES.get(self.full_module_name) or
                                self._register_new_namespace(self.full_module_name))
        
        self.logger = get_logger(self.logger_namespace)
        self.config_handler_class = config_handler_class
        
    def _register_new_namespace(self, module_name: str) -> str:
        """Register namespace baru dengan one-liner"""
        namespace = f"smartcash.ui.{module_name}" if "smartcash" not in module_name else module_name
        register_namespace(namespace, module_name.split('.')[-1].upper())
        return namespace
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan fixed handler binding"""
        try:
            suppress_all_outputs()
            
            # Create config handler
            config_handler = self._create_config_handler()
            
            # Load config
            merged_config = config or config_handler.load_config()
            
            # Create UI components
            ui_components = try_operation_safe(
                lambda: self._create_ui_components(merged_config, env, **kwargs),
                fallback_value=None
            )
            
            if not ui_components:
                return create_fallback_ui(f"Gagal membuat UI components untuk {self.module_name}", self.module_name)
            
            # Add config handler ke components
            ui_components['config_handler'] = config_handler
            
            # Setup logger dan button manager
            logger_bridge = try_operation_safe(lambda: create_ui_logger_bridge(ui_components, self.logger_namespace))
            self._add_logger_to_components(ui_components, logger_bridge)
            
            # Setup handlers dengan fixed binding
            self._setup_handlers_fixed(ui_components, merged_config, config_handler, env, **kwargs)
            
            # Validation dan finalization
            validation_result = self._validate_setup(ui_components)
            if not validation_result['valid']:
                return create_fallback_ui(validation_result['message'], self.module_name)
            
            self._finalize_setup(ui_components, merged_config)
            show_status_safe(ui_components, f"✅ {self.module_name} UI berhasil diinisialisasi", "success")
            
            return self._get_return_value(ui_components)
            
        except Exception as e:
            self.logger.error(f"❌ Error inisialisasi {self.module_name}: {str(e)}")
            return create_fallback_ui(f"Error inisialisasi: {str(e)}", self.module_name)
    
    def _create_config_handler(self) -> ConfigHandler:
        """Create config handler dengan parent module support"""
        if self.config_handler_class:
            return self.config_handler_class(self.module_name, self.parent_module)
        
        # Fallback: create BaseConfigHandler dengan extract/update dari subclass
        from smartcash.ui.handlers.config_handlers import BaseConfigHandler
        
        extract_fn = getattr(self, '_extract_config', None) if hasattr(self, '_extract_config') else None
        update_fn = getattr(self, '_update_ui', None) if hasattr(self, '_update_ui') else None
        
        return BaseConfigHandler(self.module_name, extract_fn, update_fn, self.parent_module)
    
    def _setup_handlers_fixed(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             config_handler: ConfigHandler, env=None, **kwargs) -> None:
        """Setup handlers dengan fixed button binding tanpa generator error"""
        # Setup common button handlers dengan proper function reference
        self._setup_fixed_button_handlers(ui_components, config_handler)
        
        # Setup module-specific handlers
        try_operation_safe(
            lambda: self._setup_module_handlers(ui_components, config, env, **kwargs),
            on_error=lambda e: self.logger.warning(f"⚠️ Error setup handlers: {str(e)}")
        )
        
        ui_components['config'] = config
    
    def _setup_fixed_button_handlers(self, ui_components: Dict[str, Any], config_handler: ConfigHandler) -> None:
        """Setup button handlers dengan fixed function binding - no generators"""
        
        def save_handler(button):
            """Fixed save handler tanpa generator"""
            button.disabled = True
            try:
                success = config_handler.save_config(ui_components)
                if success:
                    show_status_safe(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
                else:
                    show_status_safe(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
            except Exception as e:
                show_status_safe(ui_components, f"❌ Error save: {str(e)}", "error")
            finally:
                button.disabled = False
        
        def reset_handler(button):
            """Fixed reset handler tanpa generator"""
            button.disabled = True
            try:
                success = config_handler.reset_config(ui_components)
                if success:
                    show_status_safe(ui_components, "✅ Konfigurasi berhasil direset", "success")
                else:
                    show_status_safe(ui_components, "❌ Gagal reset konfigurasi", "error")
            except Exception as e:
                show_status_safe(ui_components, f"❌ Error reset: {str(e)}", "error")
            finally:
                button.disabled = False
        
        def cleanup_handler(button):
            """Fixed cleanup handler tanpa generator"""
            button.disabled = True
            try:
                self._do_cleanup(ui_components, button)
            except Exception as e:
                show_status_safe(ui_components, f"❌ Error cleanup: {str(e)}", "error")
            finally:
                button.disabled = False
        
        # Bind handlers dengan proper function reference
        button_mappings = {
            'save_button': save_handler,
            'reset_button': reset_handler,
            'cleanup_button': cleanup_handler
        }
        
        for button_name, handler in button_mappings.items():
            if button_name in ui_components and hasattr(ui_components[button_name], 'on_click'):
                ui_components[button_name].on_click(handler)
    
    def _do_cleanup(self, ui_components: Dict[str, Any], button) -> None:
        """Implementasi cleanup - override di subclass jika diperlukan"""
        show_status_safe(ui_components, "🧹 Membersihkan resources...", "info")
        
        # Default cleanup: clear outputs dan reset progress
        config_handler = ui_components.get('config_handler')
        if config_handler:
            config_handler._clear_ui_outputs(ui_components)
            config_handler._reset_progress_bars(ui_components)
        
        show_status_safe(ui_components, "✅ Cleanup selesai", "success")
    
    # Abstract methods - tetap sama
    @abstractmethod
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _get_critical_components(self) -> List[str]:
        pass
    
    # Helper methods dengan one-liner style
    def _add_logger_to_components(self, ui_components: Dict[str, Any], logger_bridge) -> None:
        """Tambahkan logger ke UI components dengan timestamp"""
        ui_components.update({
            'logger': logger_bridge or self.logger,
            'logger_namespace': self.logger_namespace,
            'module_name': self.module_name,
            f'{self.module_name}_initialized': True,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate setup dengan critical component check"""
        missing = [c for c in self._get_critical_components() if c not in ui_components]
        return {'valid': not missing, 'message': f"Komponen tidak ditemukan: {', '.join(missing)}" if missing else "Validasi sukses"}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup dengan metadata update"""
        ui_components.update({
            'initialized_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'initialized_by': self.__class__.__name__,
            'config': config
        })
    
    # One-liner utilities
    _get_return_value = lambda self, ui_components: ui_components.get('ui', ui_components)
    get_module_status = lambda self: {'module_name': self.module_name, 'initialized': True, 'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}


# Factory functions dengan one-liner style
def create_common_initializer(module_name: str, config_handler_class: Type[ConfigHandler] = None) -> Type[CommonInitializer]:
    """Factory untuk create CommonInitializer subclass dengan config handler"""
    class DynamicInitializer(CommonInitializer):
        def __init__(self):
            super().__init__(module_name, config_handler_class)
        
        # Placeholder methods - harus di-override
        def _create_ui_components(self, config, env=None, **kwargs):
            raise NotImplementedError("_create_ui_components must be implemented")
        
        def _setup_module_handlers(self, ui_components, config, env=None, **kwargs):
            return ui_components
        
        def _get_default_config(self):
            return {}
        
        def _get_critical_components(self):
            return ['ui']
    
    return DynamicInitializer

def register_config_handler_for_module(module_name: str, handler_class: Type[ConfigHandler]) -> None:
    """Register config handler untuk module tertentu - one-liner registry pattern"""
    globals()[f'{module_name}_config_handler'] = handler_class