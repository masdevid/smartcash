"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Initializer untuk dependency management module dengan pattern terbaru
"""

from typing import Dict, Any, Type, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logger_bridge import UILoggerBridge, create_ui_logger_bridge

class DependencyInitializer(CommonInitializer):
    """Initializer untuk dependency management dengan pattern terbaru"""
    
    def __init__(self, module_name: str = 'dependency', 
                 config_handler_class: Type[ConfigHandler] = DependencyConfigHandler,
                 **kwargs):
        """Initialize dependency initializer dengan fail-fast validation
        
        Args:
            module_name: Nama modul (default: 'dependency')
            config_handler_class: Kelas handler konfigurasi
            **kwargs: Argumen tambahan untuk parent class
        """
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
        )
        self._logger_bridge = None
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid
            
        Note:
            Jika terjadi error, akan mengembalikan dictionary dengan kunci 'error'
            yang berisi widget error untuk ditampilkan
        """
        try:
            from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
            
            # Log start of UI component creation
            self.logger.debug("Memulai pembuatan komponen UI")
            
            # Create UI components dengan error handling
            ui_components = create_dependency_main_ui(config)
            
            # Validasi tipe return
            if not isinstance(ui_components, dict):
                error_msg = f"UI components harus berupa dictionary, dapat: {type(ui_components)}"
                self.logger.error(error_msg, exc_info=True, stack_info=True)
                return self.create_error_response(
                    f"{error_msg}\n\nDetail: {str(ui_components)[:500]}"
                )
                    
            # Validasi komponen tidak kosong
            if not ui_components:
                error_msg = "UI components tidak boleh kosong"
                self.logger.error(error_msg, exc_info=True, stack_info=True)
                return self.create_error_response(error_msg)
            
            # Validasi komponen kritis
            required_components = ['ui', 'log_output', 'status_panel']
            missing = [comp for comp in required_components if comp not in ui_components]
            if missing:
                error_msg = f"Komponen UI kritis tidak ditemukan: {missing}"
                self.logger.error(
                    f"{error_msg}\nKomponen yang tersedia: {list(ui_components.keys())}",
                    exc_info=True,
                    stack_info=True
                )
                return self.create_error_response(
                    f"{error_msg}\n\nKomponen yang tersedia: {', '.join(ui_components.keys())}"
                )
            
            # Add config handler reference
            ui_components['config_handler'] = self.config_handler
            
            self.logger.debug("Pembuatan komponen UI berhasil")
            return ui_components
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"Gagal membuat komponen UI: {str(e)}\n\nTraceback:\n{error_trace}"
            self.logger.error(error_msg, exc_info=True, stack_info=True)
            return self.create_error_response(
                f"Terjadi kesalahan saat memuat antarmuka pengguna.\n\n"
                f"Error: {str(e)}\n\n"
                "Silakan periksa log untuk detail lebih lanjut.",
                e
            )
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers dengan proper logger bridge integration
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
            
        Raises:
            ValueError: Jika handler setup gagal
        """
        # Ensure logger bridge is available before setting up handlers
        if not hasattr(self, '_logger_bridge') or not self._logger_bridge:
            raise ValueError("Logger bridge belum diinisialisasi sebelum setup handlers")
                
        # Add logger bridge to ui_components untuk akses handlers
        ui_components['logger_bridge'] = self._logger_bridge
        
        from smartcash.ui.setup.dependency.handlers.event_handlers import setup_all_handlers
        
        # Setup handlers dengan error handling
        handlers = setup_all_handlers(ui_components, config, self.config_handler)
        
        if not handlers:
            raise ValueError("Gagal menginisialisasi dependency handlers")
            
        ui_components['handlers'] = handlers
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            RuntimeError: Jika gagal memuat konfigurasi default
        """
        try:
            from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
            return get_default_dependency_config()
        except Exception as e:
            raise RuntimeError(f"Gagal memuat konfigurasi default: {str(e)}") from e
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> UILoggerBridge:
        """Initialize logger bridge untuk dependency module
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Returns:
            Instance UILoggerBridge yang telah diinisialisasi
            
        Raises:
            ValueError: Jika komponen UI yang diperlukan tidak ditemukan
        """
        required_components = ['log_output', 'status_panel']
        missing = [comp for comp in required_components if comp not in ui_components]
        if missing:
            raise ValueError(f"Komponen UI yang diperlukan untuk logger bridge tidak ditemukan: {missing}")
        
        try:
            # Gunakan factory function untuk membuat logger bridge
            self._logger_bridge = create_ui_logger_bridge(
                ui_components={
                    'log_output': ui_components.get('log_output'),
                    'status_panel': ui_components.get('status_panel')
                },
                logger_name=f"{self.module_name}.ui"
            )
            return self._logger_bridge
        except Exception as e:
            raise RuntimeError(f"Gagal menginisialisasi logger bridge: {str(e)}") from e
    
    def initialize_ui(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Initialize the dependency management UI dengan error handling yang komprehensif
        
        Args:
            config: Konfigurasi opsional untuk inisialisasi
            **kwargs: Argumen tambahan
            
        Returns:
            Root UI component yang siap ditampilkan
            
        Raises:
            RuntimeError: Jika inisialisasi gagal
        """
        try:
            # Load default config jika tidak disediakan
            if config is None:
                config = self._get_default_config()
                
            # Buat komponen UI
            ui_components = self._create_ui_components(config, **kwargs)
            
            # Inisialisasi logger bridge
            self._initialize_logger_bridge(ui_components)
            
            # Setup event handlers
            ui_components = self._setup_handlers(ui_components, config, **kwargs)
            
            # Log sukses
            self._logger_bridge.info("✅ Dependency management UI berhasil diinisialisasi")
            
            # Return the root UI component
            return self._get_ui_root(ui_components)
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi dependency UI: {str(e)}"
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                self._logger_bridge.error(f"{error_msg}\n{str(e)}")
            raise RuntimeError(error_msg) from e


def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Initialize and return the dependency management UI.
    
    This is a convenience function that creates a DependencyInitializer instance
    and initializes the UI with the provided configuration.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments passed to DependencyInitializer
        
    Returns:
        The initialized UI component
        
    Example:
        >>> from smartcash.ui.setup.dependency import initialize_dependency_ui
        >>> ui = initialize_dependency_ui()
        >>> display(ui)
    """
    initializer = DependencyInitializer(**kwargs)
    return initializer.initialize_ui(config=config)