"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Initializer untuk dependency management module dengan centralized error handling
"""

from typing import Dict, Any, Type, Optional, Union
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.handlers.error_handler import create_error_response

class DependencyInitializer(CommonInitializer):
    """Initializer untuk dependency management dengan centralized error handling"""
    
    def __init__(self, module_name: str = 'dependency', 
                 config_handler_class: Type[ConfigHandler] = DependencyConfigHandler,
                 **kwargs):
        """Initialize dependency initializer dengan centralized error handling
        
        Args:
            module_name: Nama modul (default: 'dependency')
            config_handler_class: Kelas handler konfigurasi
            **kwargs: Argumen tambahan untuk parent class
        """
        try:
            super().__init__(
                module_name=module_name,
                config_handler_class=config_handler_class,
                **kwargs
            )
        except Exception as e:
            self._handle_error(f"Failed to initialize dependency initializer: {str(e)}", exc_info=True)
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan centralized error handling dan validation
        
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
                self._handle_error(error_msg, exc_info=True)
                return create_error_response(
                    f"{error_msg}\n\nDetail: {str(ui_components)[:500]}"
                )
                    
            # Validasi komponen tidak kosong
            if not ui_components:
                error_msg = "UI components tidak boleh kosong"
                self._handle_error(error_msg, exc_info=True)
                return create_error_response(error_msg)
            
            # Validasi komponen kritis untuk container-based UI
            required_components = ['container', 'main_container', 'header_container', 'form_container', 'action_container']
            missing = [comp for comp in required_components if comp not in ui_components]
            if missing:
                error_msg = f"Komponen UI kritis tidak ditemukan: {missing}"
                self._handle_error(
                    f"{error_msg}\nKomponen yang tersedia: {list(ui_components.keys())}",
                    exc_info=True
                )
                return create_error_response(
                    f"{error_msg}\n\nKomponen yang tersedia: {', '.join(ui_components.keys())}"
                )
            
            # Add config handler reference
            ui_components['config_handler'] = self.config_handler
            
            self.logger.debug("Pembuatan komponen UI berhasil")
            return ui_components
            
        except Exception as e:
            error_msg = f"Gagal membuat komponen UI: {str(e)}"
            self._handle_error(error_msg, exc_info=True)
            return create_error_response(error_msg)
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers dengan centralized error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
            
        Raises:
            ValueError: Jika handler setup gagal
        """
        try:
            from smartcash.ui.setup.dependency.handlers.event_handlers import setup_all_handlers
            
            # Setup handlers dengan centralized error handling
            handlers = setup_all_handlers(
                ui_components=ui_components,
                config=config,
                config_handler=self.config_handler
            )
            
            # Update UI components dengan handlers
            ui_components.update(handlers)
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Gagal setup handlers: {str(e)}"
            self._handle_error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration with centralized error handling
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            RuntimeError: Jika gagal memuat konfigurasi default
        """
        try:
            from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
            return get_default_dependency_config()
        except Exception as e:
            error_msg = f"Gagal memuat konfigurasi default: {str(e)}"
            self._handle_error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    # _initialize_logger_bridge method removed - using centralized error handling instead
    
    def initialize_ui(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Initialize the dependency management UI dengan centralized error handling
        
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
            
            # Setup event handlers
            ui_components = self._setup_handlers(ui_components, config, **kwargs)
            
            # Log sukses
            self.logger.info("✅ Dependency management UI berhasil diinisialisasi")
            
            # Return the root UI component
            return self._get_ui_root(ui_components)
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi dependency UI: {str(e)}"
            self._handle_error(error_msg, exc_info=True)
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