"""
Factory untuk membuat dan menampilkan modul UI Downloader.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Downloader menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class DownloaderUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Downloader.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Downloader dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_downloader_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DownloaderUIModule:
        """
        Buat instance DownloaderUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance DownloaderUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance DownloaderUIModule")
            
            # Create instance directly since DownloaderUIModule handles its own initialization
            module = DownloaderUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance DownloaderUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat DownloaderUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_downloader(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Downloader UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Downloader UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_downloader_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Downloader UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Downloader UI displayed successfully")
            else:
                logger.debug(f"✅ Downloader UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Downloader UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_downloader_display(**kwargs) -> callable:
    """
    Create a display function for the downloader UI.
    
    This is a convenience function that returns a callable that can be used
    to display the downloader UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the downloader UI
        
    Returns:
        A callable that will display the downloader UI when called
    """
    def display_fn():
        DownloaderUIFactory.create_and_display_downloader(**kwargs)
        return None
    
    return display_fn
