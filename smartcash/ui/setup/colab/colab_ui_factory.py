"""
Factory untuk membuat dan menampilkan modul UI Colab.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Colab menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
from smartcash.ui.logger import get_module_logger

class ColabUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Colab.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Colab dengan konfigurasi default yang sesuai.
    """
    
    _instance = None
    _initialized = False
    
    @classmethod
    def create_colab_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ColabUIModule:
        """
        Buat atau dapatkan instance ColabUIModule yang sudah ada.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance ColabUIModule yang sudah diinisialisasi
        """
        # Return existing instance if available
        if cls._instance is not None and cls._initialized:
            if config is not None and hasattr(cls._instance, 'update_config'):
                cls._instance.update_config(config)
            return cls._instance
            
        try:
            # Create new instance if none exists
            if cls._instance is None:
                cls._instance = ColabUIModule()
                cls._instance.log_debug("Membuat instance ColabUIModule baru")
            
            # Update config if provided
            if config is not None and hasattr(cls._instance, 'update_config'):
                cls._instance.update_config(config)
            
            # Initialize only once
            if not cls._initialized:
                cls._instance.log_debug("Menginisialisasi ColabUIModule...")
                initialization_result = cls._instance.initialize()
                if not initialization_result:
                    raise RuntimeError("Module initialization failed")
                cls._initialized = True
                cls._instance.log_debug("✅ Berhasil menginisialisasi ColabUIModule")
            
            return cls._instance
            
        except Exception as e:
            # Try to log to module first, fallback to factory logger
            error_msg = f"Gagal membuat ColabUIModule: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise
    
    @classmethod
    def create_and_display_colab(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Colab UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        
        try:
            # Buat instance modul (akan mengembalikan instance yang sudah ada jika tersedia)
            module = cls.create_colab_module(config=config, **kwargs)
            
            # Tampilkan UI menggunakan utility yang konsisten
            ui_utils.display_ui_module(
                module=module,
                module_name="Colab",
                **kwargs
            )
            # Return None explicitly to avoid displaying module object
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Colab UI: {str(e)}"
            # Try to log to module first, fallback to factory logger
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


def create_colab_display(**kwargs) -> callable:
    """
    Create a display function for the colab UI.
    
    This is a convenience function that returns a callable that can be used
    to display the colab UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the colab UI
        
    Returns:
        A callable that will display the colab UI when called
    """
    def display_fn():
        ColabUIFactory.create_and_display_colab(**kwargs)
    
    return display_fn
