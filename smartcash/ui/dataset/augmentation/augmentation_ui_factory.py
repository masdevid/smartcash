"""
Factory untuk membuat dan menampilkan modul UI Augmentation.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Augmentation menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class AugmentationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Augmentation.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Augmentation dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_augmentation_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AugmentationUIModule:
        """
        Buat instance AugmentationUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance AugmentationUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance AugmentationUIModule")
            
            # Create instance directly since AugmentationUIModule handles its own initialization
            module = AugmentationUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance AugmentationUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat AugmentationUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_augmentation(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Augmentation UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Augmentation UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_augmentation_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Augmentation UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Augmentation UI displayed successfully")
            else:
                logger.debug(f"✅ Augmentation UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Augmentation UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_augmentation_display(**kwargs) -> callable:
    """
    Create a display function for the augmentation UI.
    
    This is a convenience function that returns a callable that can be used
    to display the augmentation UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the augmentation UI
        
    Returns:
        A callable that will display the augmentation UI when called
    """
    def display_fn():
        AugmentationUIFactory.create_and_display_augmentation(**kwargs)
        return None
    
    return display_fn
