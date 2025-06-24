"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Base initializer dengan error handling yang diperbaiki dan abstract method yang jelas
"""

import datetime
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
from smartcash.common.logging import get_logger
from smartcash.common.utils import try_operation_safe, suppress_all_outputs
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.ui_utils import show_status_safe, create_fallback_ui


class CommonInitializer(ABC):
    """Base class untuk semua initializer dengan proper abstract methods dan error handling"""
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None, 
                 parent_module: Optional[str] = None):
        """Initialize dengan proper namespace dan logger setup"""
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Logger setup yang disederhanakan
        self.logger_namespace = f"smartcash.ui.{self.full_module_name}"
        self.logger = get_logger(self.logger_namespace)
        self.config_handler_class = config_handler_class
        
        self.logger.debug(f"🚀 Initializing {self.module_name} dengan parent: {parent_module}")
    
    def initialize(self, env=None, config=None, **kwargs) -> Any:
        """Main initialization dengan proper error handling dan fallback"""
        try:
            suppress_all_outputs()
            self.logger.info(f"🔄 Memulai inisialisasi {self.module_name}...")
            
            # Create config handler
            config_handler = self._create_config_handler()
            
            # Load merged config
            merged_config = config or config_handler.load_config()
            self.logger.debug(f"📄 Config loaded: {merged_config is not None}")
            
            # Create UI components - WAJIB diimplementasi oleh subclass
            ui_components = self._create_ui_components(merged_config, env, **kwargs)
            
            if not ui_components or not isinstance(ui_components, dict):
                error_msg = f"❌ {self.module_name}: _create_ui_components mengembalikan nilai invalid"
                self.logger.error(error_msg)
                return create_fallback_ui(error_msg, self.module_name)
            
            # Add config handler dan logger bridge
            ui_components['config_handler'] = config_handler
            self._add_logger_bridge(ui_components)
            
            # Setup module handlers
            result_components = self._setup_module_handlers(ui_components, merged_config, env, **kwargs)
            
            # Validation
            if not self._validate_critical_components(result_components):
                error_msg = f"❌ {self.module_name}: Komponen kritis tidak valid"
                self.logger.error(error_msg)
                return create_fallback_ui(error_msg, self.module_name)
            
            # Finalization
            self._finalize_setup(result_components, merged_config)
            
            # Post-initialization hook
            self._post_initialization_hook(result_components, merged_config, env, **kwargs)
            
            show_status_safe(f"✅ {self.module_name} berhasil diinisialisasi", "success", result_components)
            self.logger.success(f"🎉 {self.module_name} initialization selesai")
            
            return self._get_return_value(result_components)
            
        except Exception as e:
            error_msg = f"❌ Critical error dalam {self.module_name}: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return create_fallback_ui(error_msg, self.module_name)
    
    # ABSTRACT METHODS - WAJIB diimplementasi oleh subclass
    @abstractmethod
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Buat komponen UI untuk module ini.
        
        Args:
            config: Konfigurasi yang sudah dimuat
            env: Environment info (misal: 'colab')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi komponen UI yang diperlukan
            
        Raises:
            NotImplementedError: Jika tidak diimplementasi
        """
        raise NotImplementedError(f"{self.__class__.__name__} harus mengimplementasi _create_ui_components")
    
    @abstractmethod
    def _get_critical_components(self) -> List[str]:
        """
        Mengembalikan list nama komponen yang kritis untuk module ini.
        
        Returns:
            List komponen yang harus ada (misal: ['main_button', 'status_panel'])
        """
        raise NotImplementedError(f"{self.__class__.__name__} harus mengimplementasi _get_critical_components")
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Mengembalikan konfigurasi default untuk module ini.
        HARUS menggunakan function dari handlers/defaults.py
        
        Example implementation:
            from .handlers.defaults import get_default_module_config
            return get_default_module_config()
        
        Returns:
            Dict berisi konfigurasi default
        """
        raise NotImplementedError(f"{self.__class__.__name__} harus mengimplementasi _get_default_config")
    
    # OPTIONAL METHOD - bisa di-override jika diperlukan
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers spesifik untuk module ini.
        Default implementation mengembalikan ui_components tanpa perubahan.
        
        Args:
            ui_components: Komponen UI yang sudah dibuat
            config: Konfigurasi
            env: Environment info
            **kwargs: Parameter tambahan
            
        Returns:
            Dict ui_components yang sudah di-setup
        """
        return ui_components
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                                env=None, **kwargs) -> None:
        """
        Hook yang dipanggil setelah initialization selesai.
        Default implementation kosong.
        """
        pass
    
    # PRIVATE METHODS - untuk internal use
    def _create_config_handler(self) -> ConfigHandler:
        """Create config handler dengan proper error handling"""
        try:
            if self.config_handler_class:
                return self.config_handler_class(self.module_name)
            else:
                # Fallback ke default ConfigHandler
                return ConfigHandler(self.module_name)
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating config handler: {str(e)}, menggunakan default")
            return ConfigHandler(self.module_name)
    
    def _add_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Add logger bridge untuk UI logging"""
        try:
            from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
            logger_bridge = create_ui_logger_bridge(ui_components, self.logger_namespace)
            ui_components['logger_bridge'] = logger_bridge
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating logger bridge: {str(e)}")
    
    def _validate_critical_components(self, ui_components: Dict[str, Any]) -> bool:
        """Validate bahwa komponen kritis ada"""
        if not ui_components:
            return False
        
        critical_components = self._get_critical_components()
        missing = [comp for comp in critical_components if comp not in ui_components]
        
        if missing:
            self.logger.error(f"❌ Missing critical components: {', '.join(missing)}")
            return False
        
        return True
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup dengan metadata"""
        ui_components.update({
            'initialized_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'initialized_by': self.__class__.__name__,
            'module_name': self.module_name,
            'config': config
        })
    
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Any:
        """Get return value - bisa di-override untuk custom return"""
        return ui_components.get('main_container', ui_components)
    
    # UTILITY METHODS
    def get_module_status(self) -> Dict[str, Any]:
        """Get status modul ini"""
        return {
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'initialized': True,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# FACTORY FUNCTIONS
def create_common_initializer(module_name: str, 
                            ui_components_func: callable,
                            critical_components: List[str],
                            default_config: Dict[str, Any],
                            config_handler_class: Type[ConfigHandler] = None,
                            setup_handlers_func: callable = None) -> Type[CommonInitializer]:
    """
    Factory untuk membuat CommonInitializer subclass secara dynamic.
    
    Args:
        module_name: Nama module
        ui_components_func: Function untuk create UI components
        critical_components: List komponen kritis
        default_config: Konfigurasi default
        config_handler_class: Class config handler (optional)
        setup_handlers_func: Function untuk setup handlers (optional)
    
    Returns:
        CommonInitializer subclass yang siap digunakan
    """
    class DynamicInitializer(CommonInitializer):
        def __init__(self, parent_module: Optional[str] = None):
            super().__init__(module_name, config_handler_class, parent_module)
        
        def _create_ui_components(self, config, env=None, **kwargs):
            return ui_components_func(config, env, **kwargs)
        
        def _get_critical_components(self):
            return critical_components
        
        def _get_default_config(self):
            return default_config
        
        def _setup_module_handlers(self, ui_components, config, env=None, **kwargs):
            if setup_handlers_func:
                return setup_handlers_func(ui_components, config, env, **kwargs)
            return super()._setup_module_handlers(ui_components, config, env, **kwargs)
    
    DynamicInitializer.__name__ = f"{module_name.capitalize()}Initializer"
    return DynamicInitializer