"""
File: smartcash/ui/initializers/common_initializer.py
Deskripsi: Base initializer dengan fix untuk merged_config_result error
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
            
            # Load merged config dengan proper error handling
            try:
                merged_config = config or config_handler.load_config()
                if not merged_config:  # If config is empty or None
                    raise ValueError("Empty config returned")
                
                self.logger.debug(f"📄 Config loaded successfully: {bool(merged_config)}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal load config untuk {self.module_name}, menggunakan default config. Error: {str(e)}")
                merged_config = self._get_default_config()
            
            # Create UI components - WAJIB diimplementasi oleh subclass
            ui_result = try_operation_safe(
                lambda: self._create_ui_components(merged_config, env, **kwargs),
                operation_name=f"Creating UI components untuk {self.module_name}",
                logger=self.logger
            )
            
            if not ui_result or not isinstance(ui_result, dict):
                error_msg = f"❌ {self.module_name}: Gagal membuat UI components"
                self.logger.error(error_msg)
                return create_fallback_ui(error_msg, self.module_name)
            
            ui_components = ui_result
            
            # Add config handler dan logger bridge
            ui_components['config_handler'] = config_handler
            self._add_logger_bridge(ui_components)
            
            # Setup module handlers menggunakan try_operation_safe
            handlers_result = try_operation_safe(
                lambda: self._setup_module_handlers(ui_components, merged_config, env, **kwargs),
                fallback_value=ui_components,  # Return original ui_components jika gagal
                operation_name=f"Setting up handlers untuk {self.module_name}",
                logger=self.logger
            )
            
            result_components = handlers_result if handlers_result else ui_components
            
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
                operation_name=f"Post initialization untuk {self.module_name}",
                logger=self.logger
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
        # Return UI container utama atau fallback message
        if 'ui' in ui_components:
            return ui_components['ui']
        elif 'container' in ui_components:
            return ui_components['container']
        else:
            # Return first widget-like component found
            for key, value in ui_components.items():
                if hasattr(value, 'layout'):  # Likely a widget
                    return value
            
            # Final fallback
            return create_fallback_ui(f"No UI container found in {self.module_name}", self.module_name)
    
    # HELPER METHODS
    def _create_config_handler(self) -> ConfigHandler:
        """Create config handler instance atau return default"""
        if self.config_handler_class:
            try:
                return self.config_handler_class()
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal create config handler {self.config_handler_class.__name__}: {str(e)}")
        
        # Return BaseConfigHandler sebagai fallback dengan empty implementations
        from smartcash.ui.handlers.config_handlers import BaseConfigHandler
        return BaseConfigHandler(
            module_name=self.module_name,
            extract_fn=lambda ui_components: {},  # Default empty config
            update_fn=lambda ui_components, config: None,  # No-op update
            parent_module=self.parent_module
        )
    
    def _add_logger_bridge(self, ui_components: Dict[str, Any]) -> None:
        """Add logger bridge ke UI components"""
        try:
            # Simple logger bridge yang safe
            def log_to_ui(message: str, level: str = 'info'):
                try:
                    if 'log_output' in ui_components and ui_components['log_output']:
                        with ui_components['log_output']:
                            icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
                            print(f"{icons.get(level, 'ℹ️')} {message}")
                except Exception:
                    pass  # Silent fail untuk logger bridge
            
            ui_components['logger_bridge'] = log_to_ui
            ui_components['logger_namespace'] = self.logger_namespace
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal setup logger bridge: {str(e)}")
    
    def _validate_critical_components(self, ui_components: Dict[str, Any]) -> bool:
        """Validate bahwa komponen kritis tersedia"""
        try:
            critical_components = self._get_critical_components()
            missing_components = [comp for comp in critical_components if comp not in ui_components or ui_components[comp] is None]
            
            if missing_components:
                self.logger.error(f"❌ Missing critical components: {missing_components}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating components: {str(e)}")
            return False
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup dengan metadata"""
        try:
            # Tambah metadata ke ui_components
            ui_components.update({
                'module_name': self.module_name,
                'parent_module': self.parent_module,
                'initialization_time': datetime.datetime.now().isoformat(),
                'config_loaded': bool(config),
                'logger_namespace': self.logger_namespace
            })
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during finalization: {str(e)}")