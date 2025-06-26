"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer yang mengimplementasikan CommonInitializer
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler

class DownloaderInitializer(CommonInitializer):
    """
    Initializer untuk modul downloader yang mengimplementasikan CommonInitializer.
    
    Menangani inisialisasi UI downloader dan integrasi dengan backend service.
    """
    
    def __init__(self, parent_module: str = 'dataset'):
        """
        Inisialisasi downloader initializer.
        
        Args:
            parent_module: Nama modul induk (default: 'dataset')
        """
        self.parent_module = parent_module
        super().__init__(
            module_name=f"{parent_module}.downloader",
            config_handler_class=DownloaderConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Buat komponen UI untuk downloader.
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan (mengandung 'env' untuk environment)
            
        Returns:
            Dictionary berisi komponen UI
        """
        from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
        
        ui_components = create_downloader_main_ui(config)
        
        # Tambahkan metadata dan konfigurasi ke UI components
        ui_components.update({
            'module_name': self.module_name.split('.')[-1],
            'parent_module': self.parent_module,
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'target_dir': config.get('download', {}).get('target_dir', 'data'),
            'env': kwargs.get('env')
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup event handlers untuk komponen UI.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
        """
        from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
        return setup_download_handlers(ui_components, config, kwargs.get('env'))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi default untuk modul downloader.
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()
    

def initialize_downloader_ui(env: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Inisialisasi UI untuk modul downloader.
    
    Args:
        env: Konfigurasi environment opsional
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Komponen UI utama
    """
    return DownloaderInitializer().initialize(env=env, config=config, **kwargs)