"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Clean CommonInitializer dengan pure ConfigHandler dependency
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
    """Clean CommonInitializer dengan pure ConfigHandler dependency"""
    
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
        """Main initialization dengan pure ConfigHandler approach"""
        try:
            suppress_all_outputs()
            
            # Create config handler (single source of truth untuk config)
            config_handler = self._create_config_handler()
            
            # Load config menggunakan handler
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
            
            try_operation_safe(
                lambda: ui_components.update({'button_state_manager': get_button_state_manager(ui_components)}),
                on_error=lambda e: self.logger.warning(f"⚠️ Error inisialisasi button state manager: {str(e)}")
            )
            
            # Setup handlers dengan config handler
            self._setup_handlers_with_config_handler(ui_components, merged_config, config_handler, env, **kwargs)
            
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
    
    def _setup_handlers_with_config_handler(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                                           config_handler: ConfigHandler, env=None, **kwargs) -> None:
        """Setup handlers dengan ConfigHandler integration"""
        # Setup common button handlers dengan config handler
        self._setup_config_button_handlers(ui_components, config_handler)
        
        # Setup module-specific handlers
        try_operation_safe(
            lambda: self._setup_module_handlers(ui_components, config, env, **kwargs),
            on_error=lambda e: self.logger.warning(f"⚠️ Error setup handlers: {str(e)}")
        )
        
        ui_components['config'] = config
    
    def _setup_config_button_handlers(self, ui_components: Dict[str, Any], config_handler: ConfigHandler) -> None:
        """Setup button handlers menggunakan ConfigHandler dengan one-liner style"""
        button_handlers = {
            'save_button': lambda b: self._handle_save_with_config_handler(ui_components, config_handler, b),
            'reset_button': lambda b: self._handle_reset_with_config_handler(ui_components, config_handler, b),
            'cleanup_button': lambda b: self._handle_cleanup_button(ui_components, b)
        }
        
        # Bind handlers dengan button state management
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
    
    def _handle_save_with_config_handler(self, ui_components: Dict[str, Any], 
                                     config_handler: ConfigHandler, button) -> None:
        self._handle_config_action(ui_components, config_handler.save_config, button, 'save_config')

    def _handle_reset_with_config_handler(self, ui_components: Dict[str, Any], 
                                      config_handler: ConfigHandler, button) -> None:
        self._handle_config_action(ui_components, config_handler.reset_config, button, 'reset_config')

    def _handle_config_action(self, ui: Dict[str, Any], method, btn, ctx: str) -> None:
        mgr = ui.get('button_state_manager')
        (setattr(btn, 'disabled', True), method(ui), setattr(btn, 'disabled', False)) \
            if not mgr else [method(ui) for _ in [mgr.config_context(ctx).__enter__()]][0] or mgr.config_context(ctx).__exit__(None, None, None)

    def _handle_cleanup_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default cleanup handler dengan button state management"""
        button_state_manager = ui_components.get('button_state_manager')
        
        if button_state_manager:
            with button_state_manager.operation_context('cleanup'):
                self._do_cleanup(ui_components, button)
        else:
            button.disabled = True
            try:
                self._do_cleanup(ui_components, button)
            finally:
                button.disabled = False
    
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