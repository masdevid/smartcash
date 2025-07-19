"""
Factory untuk membuat dan menampilkan modul UI Visualization.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Visualization menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.visualization.visualization_uimodule import VisualizationUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class VisualizationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Visualization.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Visualization dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_visualization_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> VisualizationUIModule:
        """
        Buat instance VisualizationUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance VisualizationUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance VisualizationUIModule")
            
            # Create instance directly since VisualizationUIModule handles its own initialization
            module = VisualizationUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance VisualizationUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat VisualizationUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_visualization(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Visualization UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Visualization UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_visualization_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Visualization UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Visualization UI displayed successfully")
            else:
                logger.debug(f"✅ Visualization UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Visualization UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_visualization_display(**kwargs) -> callable:
    """
    Create a display function for the visualization UI.
    
    This is a convenience function that returns a callable that can be used
    to display the visualization UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the visualization UI
        
    Returns:
        A callable that will display the visualization UI when called
    """
    def display_fn():
        VisualizationUIFactory.create_and_display_visualization(**kwargs)
        return None
    
    return display_fn
