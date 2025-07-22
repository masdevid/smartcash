"""
Factory untuk membuat dan menampilkan modul UI Augmentation.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Augmentation menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function
from smartcash.ui.logger import get_module_logger

class AugmentationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Augmentation.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Augmentation dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> AugmentationUIModule:
        """
        Create a new instance of AugmentationUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New AugmentationUIModule instance
        """
        module = AugmentationUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_augmentation_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> AugmentationUIModule:
        """
        Buat instance AugmentationUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance AugmentationUIModule yang sudah diinisialisasi dengan caching
        """
        return create_ui_factory_method(
            module_class=AugmentationUIModule,
            module_name="Augmentation",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
                - force_refresh: Boolean, apakah akan memaksa refresh cache (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_augmentation_module',
            module_name='Augmentation',
            config=config,
            **kwargs
        )
        return display_fn()


def create_augmentation_display(**kwargs):
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
        """Display the augmentation UI with the configured settings."""
        AugmentationUIFactory.create_and_display_augmentation(**kwargs)
    
    return display_fn
