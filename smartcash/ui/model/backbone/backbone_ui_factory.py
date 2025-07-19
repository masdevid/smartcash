"""
Factory untuk membuat dan menampilkan modul UI Backbone.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Backbone menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
from smartcash.ui.logger import get_module_logger

logger = get_module_logger(__name__)

class BackboneUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Backbone.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Backbone dengan konfigurasi default yang sesuai.
    """
    
    @classmethod
    def create_backbone_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BackboneUIModule:
        """
        Buat instance BackboneUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance BackboneUIModule yang sudah diinisialisasi
        """
        try:
            logger.debug(f"Membuat instance BackboneUIModule")
            
            # Create instance directly since BackboneUIModule handles its own initialization
            module = BackboneUIModule()
            
            # Initialize with config if provided
            if config is not None:
                module.initialize(config=config, **kwargs)
            else:
                module.initialize(**kwargs)
            
            logger.debug(f"✅ Berhasil membuat instance BackboneUIModule")
            return module
            
        except Exception as e:
            logger.error(f"Gagal membuat BackboneUIModule: {e}", exc_info=True)
            raise
    
    @classmethod
    def create_and_display_backbone(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Backbone UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        try:
            logger.debug(f"Membuat dan menampilkan Backbone UI")
            
            # Get auto_display flag from kwargs (default to True if not specified)
            auto_display = kwargs.pop('auto_display', True)
            
            # Buat instance modul
            module = cls.create_backbone_module(config=config, **kwargs)
            
            # Display the UI if auto_display is True
            if auto_display:
                logger.debug(f"Displaying Backbone UI...")
                display_result = module.display_ui()
                if not display_result.get('success', False):
                    error_msg = display_result.get('message', 'Gagal menampilkan UI')
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                logger.debug(f"✅ Backbone UI displayed successfully")
            else:
                logger.debug(f"✅ Backbone UI module created (auto-display disabled)")
            
            # Use IPython.display.display() instead of returning the module
            from IPython.display import display
            display(module)
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Backbone UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise


def create_backbone_display(**kwargs) -> callable:
    """
    Create a display function for the backbone UI.
    
    This is a convenience function that returns a callable that can be used
    to display the backbone UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the backbone UI
        
    Returns:
        A callable that will display the backbone UI when called
    """
    def display_fn():
        BackboneUIFactory.create_and_display_backbone(**kwargs)
        return None
    
    return display_fn
