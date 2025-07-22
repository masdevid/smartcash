"""
Factory untuk membuat dan menampilkan modul UI Preprocessing.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Preprocessing menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function
from smartcash.ui.logger import get_module_logger

class PreprocessingUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Preprocessing.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Preprocessing dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> PreprocessingUIModule:
        """
        Create a new instance of PreprocessingUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                - enable_environment: Boolean, whether to enable environment configuration (default: True)
                
        Returns:
            New PreprocessingUIModule instance
        """
        enable_environment = kwargs.get('enable_environment', True)
        module = PreprocessingUIModule(enable_environment=enable_environment)
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_preprocessing_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> PreprocessingUIModule:
        """
        Buat instance PreprocessingUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - enable_environment: Boolean, whether to enable environment configuration (default: True)
                
        Returns:
            Instance PreprocessingUIModule yang sudah diinisialisasi dengan caching
        """
        return create_ui_factory_method(
            module_class=PreprocessingUIModule,
            module_name="Preprocessing",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
                - force_refresh: Boolean, apakah akan memaksa refresh cache (default: False)
                - enable_environment: Boolean, whether to enable environment configuration (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_preprocessing_module',
            module_name='Preprocessing',
            config=config,
            **kwargs
        )
        return display_fn()


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
