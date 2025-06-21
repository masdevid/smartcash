"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Fixed CommonInitializer dengan simplified button handling dan fixed import errors
"""

from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
import datetime

from smartcash.ui.utils.fallback_utils import create_fallback_ui, try_operation_safe, show_status_safe
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge, get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.ui_logger_namespace import KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import ConfigHandler

class CommonInitializer(ABC):
    """Fixed CommonInitializer dengan simplified button handling dan fixed imports"""
    
    def __init__(self, module_name: str, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Setup logger namespace - simplified without register_namespace
        self.logger_namespace = (KNOWN_NAMESPACES.get(f"smartcash.ui.{self.full_module_name}") or 
                                KNOWN_NAMESPACES.get(self.full_module_name) or
                                f"smartcash.ui.{self.full_module_name}")
        
        self.logger = get_logger(self.logger_namespace)
        self.config_handler_class = config_handler_class
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan simplified error handling"""
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
            
            # Setup logger 
            logger_bridge = try_operation_safe(lambda: create_ui_logger_bridge(ui_components, self.logger_namespace))
            self._add_logger_to_components(ui_components, logger_bridge)
            
            # Setup handlers dengan simplified approach
            self._setup_handlers_with_config_handler(ui_components, merged_config, config_handler, env, **kwargs)
            
            # Validation dan finalization
            validation_result = self._validate_setup(ui_components)
            if not validation_result['valid']:
                return create_fallback_ui(validation_result['message'], self.module_name)
            
            self._finalize_setup(ui_components, merged_config)
            show_status_safe(f"✅ {self.module_name} UI berhasil diinisialisasi", "success", ui_components)
            
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
        """Setup handlers dengan simplified button handling"""
        # Setup simplified button handlers
        self._setup_simplified_button_handlers(ui_components, config_handler)
        
        # Setup module-specific handlers
        try_operation_safe(
            lambda: self._setup_module_handlers(ui_components, config, env, **kwargs),
            on_error=lambda e: self.logger.warning(f"⚠️ Error setup handlers: {str(e)}")
        )
        
        ui_components['config'] = config
    
    def _setup_simplified_button_handlers(self, ui_components: Dict[str, Any], config_handler: ConfigHandler) -> None:
        """Setup simplified button handlers tanpa context manager yang bermasalah"""
        
        def safe_save_handler(button):
            """Safe save handler tanpa context manager"""
            self._safe_button_operation(
                button, 
                lambda: config_handler.save_config(ui_components),
                ui_components,
                'save'
            )
        
        def safe_reset_handler(button):
            """Safe reset handler tanpa context manager"""
            self._safe_button_operation(
                button, 
                lambda: config_handler.reset_config(ui_components),
                ui_components,
                'reset'
            )
        
        # Bind handlers dengan one-liner
        button_handlers = {
            'save_button': safe_save_handler,
            'reset_button': safe_reset_handler
        }
        
        [ui_components[btn].on_click(handler) for btn, handler in button_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
    
    def _safe_button_operation(self, button, operation: callable, ui_components: Dict[str, Any], operation_name: str) -> None:
        """Safe button operation dengan simplified state management"""
        original_disabled = getattr(button, 'disabled', False)
        
        try:
            # Disable button
            button.disabled = True
            
            # Execute operation
            result = operation()
            
            # Log success
            if result:
                self.logger.info(f"✅ {operation_name.capitalize()} operation completed successfully")
            
        except Exception as e:
            error_msg = f"❌ Error during {operation_name}: {str(e)}"
            self.logger.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            
        finally:
            # Always restore button state
            button.disabled = original_disabled
    
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
        """Daftar nama komponen kritis yang harus ada"""
        return []
        
    def _create_fallback_ui(self, error_msg: str, exc_info=None) -> Dict[str, Any]:
        """Membuat fallback UI dengan error handling yang sederhana
        
        Args:
            error_msg: Pesan error yang akan ditampilkan
            exc_info: Optional exception info tuple (type, value, traceback)
            
        Returns:
            Dictionary berisi komponen UI fallback
        """
        from smartcash.ui.utils.fallback_utils import create_fallback_ui, FallbackConfig
        import traceback
        
        # Dapatkan traceback jika tersedia
        tb_str = ""
        if exc_info and len(exc_info) > 2:
            tb_str = ''.join(traceback.format_exception(*exc_info))
            
        # Buat config untuk fallback UI
        fallback_config = FallbackConfig(
            title=f"⚠️ Error in {self.module_name}",
            module_name=self.module_name,
            traceback=tb_str
        )
        
        # Buat dan kembalikan fallback UI
        return create_fallback_ui(
            error_message=error_msg,
            exc_info=exc_info,
            config=fallback_config
        )
    
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
    def _clear_existing_widgets(self) -> None:
        """Clear existing widgets untuk avoid conflicts"""
        try:
            import gc
            from IPython.display import clear_output
            # Force garbage collection
            gc.collect()
            # Clear any existing outputs
            clear_output(wait=True)
        except Exception:
            pass  # Silent fail jika clear tidak berhasil
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