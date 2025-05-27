"""
File: smartcash/ui/utils/common_initializer.py
Deskripsi: Simplified base class untuk UI initializers tanpa cache complexity
"""

from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
import ipywidgets as widgets
from IPython.display import display
import datetime
import logging

from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager


class CommonInitializer(ABC):
    """Simplified base class untuk UI initializers tanpa cache complexity"""
    
    def __init__(self, module_name: str, logger_namespace: str):
        """Initialize common initializer"""
        self.module_name = module_name
        self.logger_namespace = logger_namespace
        self.logger = get_logger(logger_namespace)
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization method yang selalu fresh"""
        try:
            self._setup_log_suppression()
            merged_config = self._get_merged_config(config)
            
            ui_components = self._create_ui_components_safe(merged_config, env, **kwargs)
            if not ui_components:
                return self._create_error_fallback_ui("Failed to create UI components")
            
            logger_bridge = self._setup_logger_bridge_safe(ui_components)
            self._enhance_components_with_logger(ui_components, logger_bridge)
            
            ui_components = self._setup_handlers_comprehensive(ui_components, merged_config, env, **kwargs)
            
            validation_result = self._validate_setup(ui_components)
            if not validation_result['valid']:
                return self._create_error_fallback_ui(validation_result['message'])
            
            self._finalize_setup(ui_components, merged_config)
            
            if logger_bridge and 'log_output' in ui_components:
                with ui_components['log_output']:
                    logger_bridge.success(f"✅ {self.module_name} UI berhasil diinisialisasi")
            
            return self._get_return_value(ui_components)
            
        except Exception as e:
            self.logger.error(f"❌ Error inisialisasi {self.module_name}: {str(e)}")
            return self._create_error_fallback_ui(f"Initialization error: {str(e)}")
    
    # Abstract methods
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
    
    # Implementation methods
    def _get_merged_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get merged configuration tanpa cache"""
        try:
            config_manager = get_config_manager()
            saved_config = config_manager.get_config(self.module_name) if hasattr(config_manager, 'get_config') else {}
            
            merged_config = self._get_default_config()
            merged_config.update(saved_config or {})
            merged_config.update(config or {})
            
            return merged_config
        except Exception as e:
            self.logger.warning(f"⚠️ Error merging config, using default: {str(e)}")
            return self._get_default_config()
    
    _create_ui_components_safe = lambda self, config, env=None, **kwargs: self._try_with_fallback(lambda: self._create_ui_components(config, env, **kwargs))
    _setup_logger_bridge_safe = lambda self, ui_components: self._try_with_fallback(lambda: create_ui_logger_bridge(ui_components, self.logger_namespace))
    
    def _enhance_components_with_logger(self, ui_components: Dict[str, Any], logger_bridge) -> None:
        """Enhance UI components dengan logger dan metadata"""
        ui_components.update({
            'logger': logger_bridge or self.logger,
            'logger_namespace': self.logger_namespace,
            'module_name': self.module_name,
            f'{self.module_name}_initialized': True
        })
    
    def _setup_handlers_comprehensive(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan comprehensive error recovery"""
        # Button state manager
        ui_components['button_state_manager'] = self._try_with_fallback(lambda: get_button_state_manager(ui_components))
        
        # Module handlers
        ui_components = self._try_with_fallback(lambda: self._setup_module_handlers(ui_components, config, env, **kwargs)) or ui_components
        
        # Common button handlers
        self._setup_common_button_handlers(ui_components)
        
        ui_components['config'] = config
        return ui_components
    
    def _setup_common_button_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup common button handlers dengan safe checking"""
        logger = ui_components.get('logger', self.logger)
        
        common_buttons = {
            'reset_button': self._handle_reset_button,
            'save_button': self._handle_save_button,
            'cleanup_button': self._handle_cleanup_button
        }
        
        for button_key, handler in common_buttons.items():
            button = ui_components.get(button_key)
            if button and hasattr(button, 'on_click'):
                button.on_click(lambda b, h=handler: h(ui_components, b))
                logger.debug(f"✅ {button_key} handler registered")
    
    def _validate_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate setup dan critical components"""
        critical_components = self._get_critical_components()
        missing_critical = [comp for comp in critical_components if comp not in ui_components]
        
        if missing_critical:
            return {'valid': False, 'message': f"Critical components missing: {', '.join(missing_critical)}"}
        
        additional_validation = self._additional_validation(ui_components)
        return additional_validation if not additional_validation.get('valid', True) else {'valid': True, 'message': 'Setup validation passed'}
    
    _additional_validation = lambda self, ui_components: {'valid': True}
    _finalize_setup = lambda self, ui_components, config: ui_components.update({'module_initialized': True, 'initialization_timestamp': self._get_timestamp(), 'config': config})
    _get_return_value = lambda self, ui_components: ui_components.get('ui', ui_components)
    
    def _setup_log_suppression(self) -> None:
        """Setup log suppression untuk clean initialization"""
        suppression_targets = [
            'smartcash.common.environment', 'smartcash.common.config.manager', 'smartcash.common.logger',
            'smartcash.ui.utils.logger_bridge', 'requests', 'urllib3', 'http.client',
            'ipywidgets', 'traitlets'
        ]
        
        for logger_name in suppression_targets:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
    
    def _create_error_fallback_ui(self, error_message: str):
        """Create error fallback UI dengan actionable information"""
        error_html = f"""
        <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                    border-radius: 8px; color: #856404; margin: 10px 0; max-width: 800px;">
            <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi {self.module_name}</h4>
            <div style="margin: 15px 0;">
                <strong>Error Detail:</strong><br>
                <code style="background: #f8f9fa; padding: 5px; border-radius: 3px; font-size: 12px;">
                    {error_message}
                </code>
            </div>
            <div style="margin: 15px 0;">
                <strong>🔧 Solusi yang Bisa Dicoba:</strong>
                <ol style="margin: 10px 0; padding-left: 20px;">
                    <li>Restart kernel dan jalankan ulang cell</li>
                    <li>Clear output semua cell dan jalankan dari awal</li>
                    <li>Periksa koneksi internet dan dependencies</li>
                    <li>Pastikan tidak ada error pada cell-cell sebelumnya</li>
                </ol>
            </div>
        </div>
        """
        return widgets.HTML(error_html)
    
    # Helper methods
    _get_timestamp = lambda self: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _try_with_fallback = lambda self, operation, fallback=None: (lambda: operation())() if True else fallback
    
    # Default button handlers (can be overridden)
    _handle_reset_button = lambda self, ui_components, button: ui_components.get('logger', self.logger).info("🔄 Reset button clicked")
    _handle_save_button = lambda self, ui_components, button: ui_components.get('logger', self.logger).info("💾 Save button clicked")
    _handle_cleanup_button = lambda self, ui_components, button: ui_components.get('logger', self.logger).info("🧹 Cleanup button clicked")
    
    # Public utility methods
    get_module_status = lambda self: {'module_name': self.module_name, 'initialized': True, 'timestamp': self._get_timestamp()}

    def _try_with_fallback(self, operation, fallback=None):
        """Helper untuk try operation dengan fallback"""
        try:
            return operation()
        except Exception as e:
            self.logger.debug(f"Operation failed: {str(e)}")
            return fallback