"""
Factory untuk membuat dan menampilkan modul UI Preprocessing.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Preprocessing menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class PreprocessingUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Preprocessing.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Preprocessing dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_preprocessing_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PreprocessingUIModule:
        """
        Buat instance PreprocessingUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance PreprocessingUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance PreprocessingUIModule")
            
            # Create instance directly since PreprocessingUIModule handles its own initialization
            module = PreprocessingUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance PreprocessingUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat PreprocessingUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_preprocessing(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Preprocessing UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Preprocessing UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_preprocessing_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Preprocessing UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Preprocessing UI displayed successfully")
            else:
                logger.debug(f"✅ Preprocessing UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Preprocessing UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_preprocessing_display(**kwargs) -> callable:
    """
    Create a display function for the preprocessing UI.
    
    This is a convenience function that returns a callable that can be used
    to display the preprocessing UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the preprocessing UI
        
    Returns:
        A callable that will display the preprocessing UI when called
    """
    def display_fn():
        PreprocessingUIFactory.create_and_display_preprocessing(**kwargs)
        return None
    
    return display_fn
