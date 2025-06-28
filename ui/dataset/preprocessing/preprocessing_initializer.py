"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Preprocessing initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

import traceback
from typing import Dict, Any, Optional, Type, Callable
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger

class PreprocessingInitializer(CommonInitializer):
    """Enhanced PreprocessingInitializer with proper error handling"""
    
    def __init__(self, config_handler_class: Type[ConfigHandler] = PreprocessingConfigHandler):
        """Initialize preprocessing initializer with proper configuration
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to PreprocessingConfigHandler)
        """
        super().__init__(module_name='preprocessing', config_handler_class=config_handler_class)
    
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
        from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
        
        # Create UI components dengan immediate validation
        ui_components = create_preprocessing_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError(f"UI components harus berupa dictionary, dapat: {type(ui_components)}")
                
        if not ui_components:
            raise ValueError("UI components tidak boleh kosong")
        
        # Validate critical components exist
        required_components = ['ui', 'log_output', 'status_panel']
        missing = [comp for comp in required_components if comp not in ui_components]
        if missing:
            raise ValueError(f"Komponen UI kritis tidak ditemukan: {missing}")
        
        # Add config handler reference
        ui_components['config_handler'] = self.config_handler
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers dengan proper error handling dan initializer connection"""
        from smartcash.ui.dataset.preprocessing.handlers.event_handlers import setup_all_handlers
        
        try:
            # Setup all handlers using the centralized setup function
            handlers = setup_all_handlers(ui_components, config, self.config_handler)
            
            # Connect handlers to this initializer for error propagation
            for handler in handlers.values():
                if hasattr(handler, 'setup_with_initializer'):
                    handler.setup_with_initializer(self)
            
            # Store handlers for later use
            self._handlers = handlers
            
            # Store UI components reference for error handling
            self._ui_components = ui_components
            
        except Exception as e:
            error_msg = "Failed to initialize handlers"
            self.logger.error(f"{error_msg}: {str(e)}", exc_info=True)
            # Re-raise to let the parent class handle it
            raise RuntimeError(f"{error_msg}: {str(e)}") from e
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback handling
        
        Returns:
            Dictionary berisi konfigurasi default yang valid
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except ImportError as e:
            self.logger.error(f"❌ Gagal import default config: {str(e)}")
            raise ImportError(f"Default preprocessing config tidak ditemukan: {str(e)}") from e
    
    def _pre_initialize_checks(self, **kwargs) -> None:
        """Pre-initialization checks untuk preprocessing requirements
        
        Args:
            **kwargs: Arguments untuk validasi
            
        Raises:
            RuntimeError: Jika dependencies tidak memenuhi syarat
        """
        # Check critical imports
        try:
            import ipywidgets
            import numpy
            from smartcash.dataset.preprocessor import api as preprocessor_api
        except ImportError as e:
            raise RuntimeError(f"Dependencies preprocessing tidak lengkap: {str(e)}") from e
        
        # Check environment compatibility
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            if not env_manager.get_dataset_path().exists():
                self.logger.warning("⚠️ Dataset path tidak ditemukan, akan dibuat otomatis")
        except Exception as e:
            self.logger.warning(f"⚠️ Environment check warning: {str(e)}")
    
    def _after_init_checks(self, ui_components: Dict[str, Any], **kwargs) -> None:
        """Post-initialization validation dan health checks
        
        Args:
            ui_components: Komponen UI yang telah diinisialisasi
            **kwargs: Arguments tambahan
            
        Raises:
            RuntimeError: Jika post-init validation gagal
        """
        # Validate UI components integrity
        critical_widgets = ['preprocess_button', 'check_button', 'save_button']
        missing_widgets = [w for w in critical_widgets if not ui_components.get(w)]
        if missing_widgets:
            raise RuntimeError(f"Widget kritis tidak ditemukan: {missing_widgets}")
        
        # Validate handlers are properly attached
        if 'handlers' not in ui_components:
            raise RuntimeError("Event handlers tidak terpasang dengan benar")
        
        # Test logger bridge functionality
        if self._logger_bridge:
            try:
                self._logger_bridge.info("🧪 Testing logger bridge connectivity...")
            except Exception as e:
                self.logger.warning(f"⚠️ Logger bridge test warning: {str(e)}")
    

def initialize_preprocessing_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Factory function untuk inisialisasi preprocessing UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Komponen UI utama yang siap ditampilkan
    """
    initializer = PreprocessingInitializer()
    return initializer.initialize(config=config, **kwargs)