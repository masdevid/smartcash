"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan pattern terbaru dari CommonInitializer"""
    
    def __init__(self, module_name: str = 'downloader', 
                 config_handler_class: Type[ConfigHandler] = DownloaderConfigHandler,
                 **kwargs):
        """Initialize downloader initializer dengan fail-fast validation
        
        Args:
            module_name: Nama modul (default: 'downloader')
            config_handler_class: Kelas handler konfigurasi
            **kwargs: Argumen tambahan untuk parent class
        """
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
        )
    
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
        from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
        
        # Create UI components dengan immediate validation
        ui_components = create_downloader_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError(f"UI components harus berupa dictionary, dapat: {type(ui_components)}")
                
        if not ui_components:
            raise ValueError("UI components tidak boleh kosong")
        
        # Validate critical components exist
        required_components = ['ui', 'log_output', 'status_panel']
        missing = [comp for comp in required_components if comp not in ui_components]
        if missing:
            raise ValueError(f"Komponen UI kritis tidak ditemukan: {missing}")
        
        # Add metadata untuk tracking
        ui_components.update({
            'module_name': self.module_name,
            'config_handler': self.config_handler,
            'initialization_timestamp': self._get_timestamp(),
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'target_dir': config.get('download', {}).get('target_dir', 'data'),
            'env': kwargs.get('env')
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
        
        from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
        
        # Setup handlers dengan error handling
        handlers = setup_download_handlers(ui_components, config, kwargs.get('env'))
        
        if not handlers:
            raise ValueError("Gagal menginisialisasi download handlers")
            
        ui_components['handlers'] = handlers
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback handling
        
        Returns:
            Dictionary berisi konfigurasi default yang valid
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
            return get_default_downloader_config()
        except ImportError as e:
            self.logger.error(f"❌ Gagal import default config: {str(e)}")
            raise ImportError(f"Default downloader config tidak ditemukan: {str(e)}") from e
    
    def _pre_initialize_checks(self, **kwargs) -> None:
        """Pre-initialization checks untuk downloader requirements
        
        Args:
            **kwargs: Arguments untuk validasi
            
        Raises:
            RuntimeError: Jika dependencies tidak memenuhi syarat atau konfigurasi tidak valid
        """
        # Check critical imports
        try:
            import ipywidgets
            import requests
            # Downloader doesn't have an api module like preprocessing
            # Only check for required packages
        except ImportError as e:
            raise RuntimeError(f"Dependencies downloader tidak lengkap: {str(e)}") from e
        
        # Check for Roboflow API key in Colab secrets
        try:
            from google.colab import userdata  # Will raise ImportError if not in Colab
            
            # Try to get Roboflow API key from Colab secrets
            try:
                roboflow_api_key = userdata.get('ROBOFLOW_API_KEY')
                if not roboflow_api_key:
                    raise RuntimeError(
                        "Roboflow API key tidak ditemukan di Colab secrets.\n"
                        "Silakan tambahkan API key Anda di Colab secrets dengan nama 'ROBOFLOW_API_KEY'"
                    )
                # Set the API key in environment for any code that might need it
                import os
                os.environ['ROBOFLOW_API_KEY'] = roboflow_api_key
                
            except userdata.Error as e:
                raise RuntimeError(
                    "Gagal mengakses Colab secrets. "
                    "Pastikan Anda sudah login ke akun Google Colab dan "
                    f"sudah menambahkan Roboflow API key ke Colab secrets. Error: {str(e)}"
                )
                
        except ImportError:
            # Not in Colab, check environment variable as fallback
            import os
            if not os.environ.get('ROBOFLOW_API_KEY'):
                self.logger.warning(
                    "⚠️ ROBOFLOW_API_KEY tidak ditemukan. "
                    "Silakan set environment variable ROBOFLOW_API_KEY"
                )
        
        # Check environment compatibility
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            # Use get_data_path() instead of get_downloads_path()
            if not env_manager.get_data_path().exists():
                self.logger.warning("⚠️ Data path tidak ditemukan, akan dibuat otomatis")
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
        critical_widgets = ['download_button', 'check_button', 'save_button']
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
    
    def _get_timestamp(self) -> str:
        """Get current timestamp untuk tracking"""
        from datetime import datetime
        return datetime.now().isoformat()


def initialize_downloader_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Factory function untuk inisialisasi downloader UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Komponen UI utama yang siap ditampilkan
    """
    initializer = DownloaderInitializer()
    return initializer.initialize(config=config, **kwargs)