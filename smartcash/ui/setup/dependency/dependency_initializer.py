"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Initializer untuk dependency management module dengan pattern terbaru
"""

from typing import Dict, Any, Type, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.logger_bridge import LoggerBridge

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
            
        Raises:
            ValueError: Jika UI components tidak valid atau kosong
        """
        from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
        
        # Create UI components dengan immediate validation
        ui_components = create_dependency_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError(f"UI components harus berupa dictionary, dapat: {type(ui_components)}")
                
        if not ui_components:
            raise ValueError("UI components tidak boleh kosok")
        
        # Validate critical components exist
        required_components = ['ui', 'log_output', 'status_panel']
        missing = [comp for comp in required_components if comp not in ui_components]
        if missing:
            raise ValueError(f"Komponen UI kritis tidak ditemukan: {missing}")
        
        # Add metadata untuk tracking
        ui_components.update({
            'module_name': self.module_name,
            'config_handler': self.config_handler,
            'initialization_timestamp': self._get_timestamp()
        })
        
        return ui_components
    
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
    
    def _initialize_logger_bridge(self, ui_components: Dict[str, Any]) -> LoggerBridge:
        """Initialize logger bridge untuk dependency module
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Returns:
            Instance LoggerBridge yang telah diinisialisasi
            
        Raises:
            ValueError: Jika komponen UI yang diperlukan tidak ditemukan
        """
        if 'log_output' not in ui_components or 'status_panel' not in ui_components:
            raise ValueError("Komponen UI yang diperlukan untuk logger bridge tidak ditemukan")
            
        self._logger_bridge = LoggerBridge(
            log_output=ui_components.get('log_output'),
            summary_output=ui_components.get('status_panel'),
            module_name=self.module_name
        )
        return self._logger_bridge
        
    def initialize_ui(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Initialize the dependency management UI dengan error handling yang komprehensif
        
        Args:
            config: Konfigurasi opsional untuk inisialisasi
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang telah diinisialisasi
            
        Raises:
            RuntimeError: Jika inisialisasi gagal
        """
        try:
            # Load default config jika tidak disediakan
            if config is None:
                config = self._get_default_config()
                
            # Create UI components
            ui_components = self._create_ui_components(config, **kwargs)
            
            # Initialize logger bridge
            self._initialize_logger_bridge(ui_components)
            
            # Setup handlers
            ui_components = self._setup_handlers(ui_components, config, **kwargs)
            
            # Log successful initialization
            self._logger_bridge.info("✅ Dependency management UI berhasil diinisialisasi")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal menginisialisasi dependency UI: {str(e)}"
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                self._logger_bridge.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e