"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Base initializer dengan import yang diperbaiki dan proper error handling
"""

import datetime
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List

# Import yang diperbaiki - menggunakan structure yang konsisten
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import try_operation_safe, show_status_safe, create_fallback_ui
from smartcash.ui.handlers.config_handlers import ConfigHandler


class CommonInitializer(ABC):
    """Base class untuk semua initializer dengan proper abstract methods dan error handling yang diperbaiki"""
    
    def __init__(self, module_name: str, config_handler_class: Type[ConfigHandler] = None, 
                 parent_module: Optional[str] = None):
        """Initialize dengan proper namespace dan logger setup"""
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Logger setup dengan namespace yang benar
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
            
            # Load merged config menggunakan try_operation_safe
            merged_config_result = try_operation_safe(
                lambda: config or config_handler.load_config(),
                f"Loading config untuk {self.module_name}",
                self.logger
            )
            
            if not merged_config_result.success:
                merged_config = self._get_default_config()
                self.logger.warning(f"⚠️ Menggunakan default config untuk {self.module_name}")
            else:
                merged_config = merged_config_result.data
            
            self.logger.debug(f"📄 Config loaded: {merged_config is not None}")
            
            # Create UI components - WAJIB diimplementasi oleh subclass
            ui_result = try_operation_safe(
                lambda: self._create_ui_components(merged_config, env, **kwargs),
                f"Creating UI components untuk {self.module_name}",
                self.logger
            )
            
            if not ui_result.success or not ui_result.data:
                error_msg = f"❌ {self.module_name}: Gagal membuat UI components"
                self.logger.error(error_msg)
                return create_fallback_ui(error_msg, self.module_name)
            
            ui_components = ui_result.data
            
            # Add config handler dan logger bridge
            ui_components['config_handler'] = config_handler
            self._add_logger_bridge(ui_components)
            
            # Setup module handlers menggunakan try_operation_safe
            handlers_result = try_operation_safe(
                lambda: self._setup_module_handlers(ui_components, merged_config, env, **kwargs),
                f"Setting up handlers untuk {self.module_name}",
                self.logger
            )
            
            result_components = handlers_result.data if handlers_result.success else ui_components
            
            # Validation
            if not self._validate_critical_components(result_components):
                error_msg = f"❌ {self.module_name}: Komponen kritis tidak valid"
                self.logger.error(error_msg)
                return create_fallback_ui(error_msg, self.module_name)
            
            # Finalization
            self._finalize_setup(result_components, merged_config)
            
            # Post-initialization hook
            try_operation_safe(
                lambda: self._post_initialization_hook(result_components, merged_config, env, **kwargs),
                f"Post initialization untuk {self.module_name}",
                self.logger
            )
            
            show_status_safe(f"✅ {self.module_name} berhasil diinisialisasi", "success", result_components)
            self.logger.info(f"🎉 {self.module_name} initialization selesai")
            
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
            config: Konfigurasi yang sudah di-load/merge
            env: Environment info
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi UI components yang diperlukan
        """
        pass
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration untuk module ini.
        HARUS menggunakan function dari handlers/defaults.py
        
        Returns:
            Dict berisi default configuration
        """
        pass
    
    @abstractmethod
    def _get_critical_components(self) -> List[str]:
        """
        Return list komponen kritis yang harus ada setelah initialization.
        
        Returns:
            List nama komponen kritis
        """
        pass
    
    # OPTIONAL METHODS - bisa di-override oleh subclass
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                              env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers untuk module ini.
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
    
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Any:
        """
        Determine nilai return dari initialize().
        Default implementation mengembalikan UI container atau fallback.
        """
        if 'ui' in ui_components:
            return ui_components['ui']
        elif 'container' in ui_components:
            return ui_components['container']
        else:
            # Return first widget-like object
            for key, value in ui_components.items():
                if hasattr(value, 'layout') or hasattr(value, 'children'):
                    return value
            return None
    
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
            ui_components['logger'] = self.logger
        except Exception as e:
            self.logger.warning(f"⚠️ Error creating logger bridge: {str(e)}")
            # Fallback: set logger saja
            ui_components['logger'] = self.logger
    
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