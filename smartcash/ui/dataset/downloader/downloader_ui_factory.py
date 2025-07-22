"""
Factory untuk membuat dan menampilkan modul UI Downloader.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Downloader menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function
from smartcash.ui.logger import get_module_logger

class DownloaderUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Downloader.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Downloader dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> DownloaderUIModule:
        """
        Create a new instance of DownloaderUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New DownloaderUIModule instance
        """
        module = DownloaderUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_downloader_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> DownloaderUIModule:
        """
        Buat instance DownloaderUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance DownloaderUIModule yang sudah diinisialisasi dengan caching
        """
        return create_ui_factory_method(
            module_class=DownloaderUIModule,
            module_name="Downloader",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
                - force_refresh: Boolean, apakah akan memaksa refresh cache (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_downloader_module',
            module_name='Downloader',
            config=config,
            **kwargs
        )
        return display_fn()


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
