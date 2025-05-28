"""
File: smartcash/ui/utils/common_initializer.py
Deskripsi: Refactored common initializer menggunakan consolidated utilities
"""

from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import datetime

from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.utils.logging_utils import suppress_backend_logs
from smartcash.ui.utils.fallback_utils import create_fallback_ui, try_operation_safe


class CommonInitializer(ABC):
    """Refactored common initializer menggunakan consolidated utilities"""
    
    def __init__(self, module_name: str, logger_namespace: str):
        self.module_name = module_name
        self.logger_namespace = logger_namespace
        self.logger = get_logger(logger_namespace)
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan consolidated utilities"""
        try:
            suppress_backend_logs()  # Consolidated log suppression
            merged_config = self._get_merged_config(config)
            
            ui_components = try_operation_safe(
                lambda: self._create_ui_components(merged_config, env, **kwargs),
                fallback_value=None
            )
            
            if not ui_components:
                return create_fallback_ui("Failed to create UI components", self.module_name)
            
            logger_bridge = try_operation_safe(
                lambda: create_ui_logger_bridge(ui_components, self.logger_namespace)
            )
            
            self._enhance_components_with_logger(ui_components, logger_bridge)
            ui_components = self._setup_handlers_comprehensive(ui_components, merged_config, env, **kwargs)
            
            validation_result = self._validate_setup(ui_components)
            if not validation_result['valid']:
                return create_fallback_ui(validation_result['message'], self.module_name)
            
            self._finalize_setup(ui_components, merged_config)
            
            if logger_bridge and 'log_output' in ui_components:
                with ui_components['log_output']:
                    logger_bridge.success(f"✅ {self.module_name} UI berhasil diinisialisasi")
            
            return self._get_return_value(ui_components)
            
        except Exception as e:
            self.logger.error(f"❌ Error inisialisasi {self.module_name}: {str(e)}")
            return create_fallback_ui(f"Initialization error: {str(e)}", self.module_name)
    
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
    
    def _get_merged_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get merged configuration dengan safe fallback"""
        try:
            config_manager = get_config_manager()
            saved_config = getattr(config_manager, 'get_config', lambda x: {})(self.module_name)
            
            merged_config = self._get_default_config()
            merged_config.update(saved_config or {})
            merged_config.update(config or {})
            
            return merged_config
        except Exception as e:
            self.logger.warning(f"⚠️ Error merging config, using default: {str(e)}")
            return self._get_default_config()
    
    def _enhance_components_with_logger(self, ui_components: Dict[str, Any], logger_bridge) -> None:
        """Enhance UI components dengan logger dan metadata - one-liner style"""
        ui_components.update({
            'logger': logger_bridge or self.logger,
            'logger_namespace': self.logger_namespace,
            'module_name': self.module_name,
            f'{self.module_name}_initialized': True
        })
    
    def _setup_handlers_comprehensive(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan comprehensive error recovery"""
        # Button state manager dengan safe fallback
        ui_components['button_state_manager'] = try_operation_safe(
            lambda: get_button_state_manager(ui_components)
        )
        
        # Module handlers dengan safe execution
        ui_components = try_operation_safe(
            lambda: self._setup_module_handlers(ui_components, config, env, **kwargs),
            fallback_value=ui_components
        )
        
        # Common button handlers
        self._setup_common_button_handlers(ui_components)
        
        ui_components['config'] = config
        return ui_components
    
    def _setup_common_button_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup common button handlers dengan safe checking - one-liner style"""
        logger = ui_components.get('logger', self.logger)
        
        # One-liner button handler registration
        button_handlers = {
            'reset_button': lambda ui, b: logger.info("🔄 Reset button clicked"),
            'save_button': lambda ui, b: logger.info("💾 Save button clicked"),
            'cleanup_button': lambda ui, b: logger.info("🧹 Cleanup button clicked")
        }
        
        [getattr(ui_components.get(btn_key), 'on_click', lambda x: None)(
            lambda b, handler=handler: handler(ui_components, b)
        ) for btn_key, handler in button_handlers.items() if btn_key in ui_components]
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate setup dengan one-liner critical component check"""
        critical_components = self._get_critical_components()
        missing_critical = [comp for comp in critical_components if comp not in ui_components]
        
        if missing_critical:
            return {'valid': False, 'message': f"Critical components missing: {', '.join(missing_critical)}"}
        
        additional_validation = getattr(self, '_additional_validation', lambda x: {'valid': True})(ui_components)
        return additional_validation if not additional_validation.get('valid', True) else {'valid': True, 'message': 'Setup validation passed'}
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup dengan one-liner metadata update"""
        ui_components.update({
            'module_initialized': True, 
            'initialization_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            'config': config
        })
    
    # One-liner utilities
    _get_return_value = lambda self, ui_components: ui_components.get('ui', ui_components)
    get_module_status = lambda self: {'module_name': self.module_name, 'initialized': True, 'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    # Default button handlers yang bisa di-override
    def _handle_reset_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default reset handler"""
        ui_components.get('logger', self.logger).info("🔄 Reset button clicked")
    
    def _handle_save_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default save handler"""
        ui_components.get('logger', self.logger).info("💾 Save button clicked")
    
    def _handle_cleanup_button(self, ui_components: Dict[str, Any], button) -> None:
        """Default cleanup handler"""
        ui_components.get('logger', self.logger).info("🧹 Cleanup button clicked")