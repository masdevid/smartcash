"""
Factory untuk membuat dan menampilkan modul UI Dependency.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Dependency menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class DependencyUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Dependency.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Dependency dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_dependency_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DependencyUIModule:
        """
        Buat instance DependencyUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance DependencyUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance DependencyUIModule")
            
            # Create instance directly since DependencyUIModule handles its own initialization
            module = DependencyUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance DependencyUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat DependencyUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_dependency(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Dependency UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Dependency UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_dependency_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Dependency UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Dependency UI displayed successfully")
            else:
                logger.debug(f"✅ Dependency UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Dependency UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_dependency_display(**kwargs) -> callable:
    """
    Create a display function for the dependency UI.
    
    This is a convenience function that returns a callable that can be used
    to display the dependency UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the dependency UI
        
    Returns:
        A callable that will display the dependency UI when called
    """
    def display_fn():
        DependencyUIFactory.create_and_display_dependency(**kwargs)
        return None
    
    return display_fn
