"""
Factory untuk membuat dan menampilkan modul UI Training.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Training menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.training.training_uimodule import TrainingUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function
from smartcash.ui.logger import get_module_logger

class TrainingUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Training.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Training dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    # Singleton pattern implementation
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to prevent duplication."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> TrainingUIModule:
        """
        Create a new instance of TrainingUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New TrainingUIModule instance
        """
        module = TrainingUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_training_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> TrainingUIModule:
        """
        Buat instance TrainingUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance TrainingUIModule yang sudah diinisialisasi dengan caching
        """
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            return create_ui_factory_method(
                module_class=TrainingUIModule,
                module_name="Training",
                create_module_func=cls._create_module_instance
            )(config=config, force_refresh=force_refresh, **kwargs)
            
        except Exception as e:
            # Critical errors always logged
            error_msg = f"Failed to create TrainingUIModule: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise
    
    @classmethod
    def create_and_display_training(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Training UI.
        
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
            create_method_name='create_training_module',
            module_name='Training',
            config=config,
            **kwargs
        )
        return display_fn()


def create_training_display(**kwargs) -> callable:
    """
    Create a display function for the training UI.
    
    This is a convenience function that returns a callable that can be used
    to display the training UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the training UI
        
    Returns:
        A callable that will display the training UI when called
    """
    def display_fn():
        TrainingUIFactory.create_and_display_training(**kwargs)
        return None
    
    return display_fn
