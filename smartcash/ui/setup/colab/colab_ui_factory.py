"""
Factory untuk membuat dan menampilkan modul UI Colab.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Colab menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class ColabUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Colab.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Colab dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_colab_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ColabUIModule:
        """
        Buat instance ColabUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance ColabUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance ColabUIModule")
            
            # Create instance directly since ColabUIModule handles its own initialization
            module = ColabUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance ColabUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat ColabUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_colab(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Buat dan tampilkan modul Colab UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            Dict berisi informasi modul atau error message
        """
        try:
            logger.debug(f"Membuat dan menampilkan Colab UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_colab_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Colab UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    return {'success': False, 'message': error_msg}
                logger.debug(f"✅ Colab UI displayed successfully")
            else:
                logger.debug(f"✅ Colab UI module created (auto-display disabled)")
            
            # Return the module to allow for more flexible usage
            return module
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Colab UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'message': error_msg}


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
        return ColabUIFactory.create_and_display_colab(**kwargs)
    
    return display_fn
