"""
Factory untuk membuat dan menampilkan modul UI Colab.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Colab menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.setup.colab.colab_uimodule import ColabUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function

class ColabUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Colab.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Colab dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> ColabUIModule:
        """
        Create a new instance of ColabUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New ColabUIModule instance
        """
        module = ColabUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_colab_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> ColabUIModule:
        """
        Buat instance ColabUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance ColabUIModule yang sudah diinisialisasi dengan caching
        """
        return create_ui_factory_method(
            module_class=ColabUIModule,
            module_name="Colab",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
                - force_refresh: Boolean, apakah akan memaksa refresh cache (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_colab_module',
            module_name='Colab',
            config=config,
            **kwargs
        )
        return display_fn()


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
        return None
    
    return display_fn
