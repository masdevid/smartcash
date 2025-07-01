"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import create_error_response

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan pattern terbaru dari CommonInitializer"""
    
    def __init__(self, config_handler_class: Type[ConfigHandler] = DownloaderConfigHandler):
        """Initialize downloader initializer with proper configuration
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to DownloaderConfigHandler)
        """
        super().__init__(module_name='downloader', config_handler_class=config_handler_class)
        # Parent class handles logger initialization through get_module_logger
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid dengan minimal keys:
            - 'ui': Komponen UI utama
            - 'log_output': Output log widget
            - 'status_panel': Panel status
            
        Raises:
            ValueError: Jika UI components tidak valid atau komponen penting tidak ada
        """
        try:
            from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
            
            # Ensure we have a valid config
            if not config:
                config = self.config_handler.get_default_config()
                self.logger.debug("Menggunakan konfigurasi default")
            
            # Ensure required sections exist
            if 'downloader' not in config:
                config['downloader'] = {}
                self.logger.debug("Menambahkan section 'downloader' ke konfigurasi")
                
            # Ensure basic structure exists
            if 'basic' not in config['downloader']:
                default_basic = self.config_handler.get_default_config().get('downloader', {}).get('basic', {})
                config['downloader']['basic'] = default_basic
                self.logger.debug("Menggunakan konfigurasi basic default")
                
            # Ensure advanced structure exists
            if 'advanced' not in config['downloader']:
                default_advanced = self.config_handler.get_default_config().get('downloader', {}).get('advanced', {})
                config['downloader']['advanced'] = default_advanced
                self.logger.debug("Menggunakan konfigurasi advanced default")
            
            # Create UI components dengan immediate validation
            self.logger.info("Membuat komponen UI utama")
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
                
            # Add logger to ui_components for child components
            # Logger is already added by CommonInitializer
            
            # Add module-specific metadata
            ui_components.update({
                'config_handler': self.config_handler,
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'target_dir': config.get('download', {}).get('target_dir', 'data'),
                'env': kwargs.get('env')
            })
            
            self.logger.info("Komponen UI berhasil diinisialisasi")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Gagal membuat komponen UI: {str(e)}", exc_info=True)
            raise
    
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
        self.logger.debug("Memulai setup handlers")
        
        # Ensure logger bridge is available before setting up handlers
        if not hasattr(self, '_logger') or not self._logger:
            raise ValueError("Logger bridge belum diinisialisasi sebelum setup handlers")
        
        try:
            from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
            
            # Setup handlers dengan error handling
            self.logger.info("Menginisialisasi download handlers")
            handlers = setup_download_handlers(ui_components, config, kwargs.get('env'))
            
            if not handlers:
                raise ValueError("Gagal menginisialisasi download handlers")
                
            ui_components['handlers'] = handlers
            self.logger.info("Handlers berhasil diinisialisasi")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Gagal setup handlers: {str(e)}", exc_info=True)
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan fallback handling
        
        Returns:
            Dictionary berisi konfigurasi default yang valid
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        self.logger.debug("Mengambil konfigurasi default")
        
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
            config = get_default_downloader_config()
            self.logger.debug("Konfigurasi default berhasil dimuat")
            return config
            
        except ImportError as e:
            error_msg = f"Gagal import default config: {str(e)}"
            self.logger.error(error_msg)
            raise ImportError(f"Default downloader config tidak ditemukan: {str(e)}") from e
    
    def _pre_initialize_checks(self, **kwargs) -> None:
        """Pre-initialization checks untuk downloader requirements
        
        Args:
            **kwargs: Arguments untuk validasi
            
        Raises:
            RuntimeError: Jika dependencies tidak memenuhi syarat atau konfigurasi tidak valid
        """
        self.logger.debug("Memulai pre-initialization checks")
        
        # Check critical imports
        try:
            import ipywidgets
            import requests
            self.logger.debug("Dependencies terpenuhi: ipywidgets, requests")
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
            from pathlib import Path
            
            env_manager = get_environment_manager()
            # Use data directory for downloader
            data_dir = env_manager._resolve_data_path()
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ Created data directory: {data_dir}")
                
            # Create a downloads subdirectory
            download_dir = data_dir / 'downloads'
            if not download_dir.exists():
                download_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"✅ Created downloads directory: {download_dir}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Environment check warning: {str(e)}")
            # Don't fail if there's an environment check warning
            # The downloader can still function with default paths
    
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
        
        # Logger is already added by parent class via _add_logger_bridge method
        # No need for backward compatibility with logger_bridge
        
        # Test logger bridge functionality
        try:
            ui_components['logger'].info("🧪 Testing logger bridge connectivity...")
        except Exception as e:
            ui_components['logger'].warning(f"⚠️ Logger bridge test warning: {str(e)}")
    


def initialize_downloader_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Factory function untuk inisialisasi downloader UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Widget UI utama yang siap ditampilkan atau dictionary dengan 'ui' key
        
    Example:
        ```python
        ui = initialize_downloader_ui(config=my_config)
        display(ui)  # or display(ui['ui']) if it's a dict
        ```
    """
    try:
        # Get module logger for initialization errors
        logger = get_module_logger('smartcash.ui.dataset.downloader')
        
        initializer = DownloaderInitializer()
        result = initializer.initialize(config=config, **kwargs)
        
        # Handle error response
        if isinstance(result, dict) and result.get('error'):
            return result['ui']
        return result
    except Exception as e:
        error_msg = f"❌ Gagal menginisialisasi downloader UI: {str(e)}"
        logger = get_module_logger('smartcash.ui.dataset.downloader')
        logger.error(error_msg, exc_info=True)
        # Use error_response from error_handler
        error_response = create_error_response(error_msg, e, "Downloader Error")
        return {'ui': error_response['container'], 'error': True}