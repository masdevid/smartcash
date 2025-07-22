"""
Factory untuk membuat dan menampilkan modul UI Pretrained.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Pretrained menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.pretrained.pretrained_uimodule import PretrainedUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function

class PretrainedUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Pretrained.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Pretrained dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> PretrainedUIModule:
        """
        Create a new instance of PretrainedUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New PretrainedUIModule instance
        """
        module = PretrainedUIModule()
        
        # Initialize the module with config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_pretrained_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> PretrainedUIModule:
        """
        Buat instance PretrainedUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance PretrainedUIModule yang sudah diinisialisasi dengan caching
        """
        return create_ui_factory_method(
            module_class=PretrainedUIModule,
            module_name="Pretrained",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
    @classmethod
    def create_and_display_pretrained(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Pretrained UI.
        
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
            create_method_name='create_pretrained_module',
            module_name='Pretrained',
            config=config,
            **kwargs
        )
        return display_fn()


def create_pretrained_display(**kwargs) -> callable:
    """
    Create a display function for the pretrained UI.
    
    This is a convenience function that returns a callable that can be used
    to display the pretrained UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the pretrained UI
        
    Returns:
        A callable that will display the pretrained UI when called
    """
    def display_fn():
        """Display the pretrained UI with the configured settings."""
        PretrainedUIFactory.create_and_display_pretrained(**kwargs)
        return None
    
    return display_fn
