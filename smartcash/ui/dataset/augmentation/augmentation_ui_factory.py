"""
Factory untuk membuat dan menampilkan modul UI Augmentation.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Augmentation menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule
from smartcash.ui.logger import get_module_logger

class AugmentationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Augmentation.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Augmentation dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Cache lifecycle management for component reuse
    - 📊 Singleton pattern to prevent duplication  
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    # Singleton pattern implementation
    _instance = None
    _initialized = False
    
    # Cache lifecycle management
    _component_cache = {}
    _cache_valid = False
    
    def __new__(cls):
        """Singleton pattern to prevent duplication."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
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
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            # Cache lifecycle management - Creation phase
            cache_key = f"augmentation_module_{hash(str(config))}"
            
            # Check cache validity
            if not force_refresh and instance._cache_valid and cache_key in instance._component_cache:
                cached_module = instance._component_cache[cache_key]
                if cached_module and hasattr(cached_module, '_is_initialized') and cached_module._is_initialized:
                    # Cache hit - return cached instance
                    return cached_module
            
            # Cache miss or invalid - create new instance
            module = AugmentationUIModule()
            
            # Minimal logging for performance
            if config is not None and hasattr(module, 'update_config'):
                module.update_config(config)
            
            # Initialize with validation
            initialization_result = module.initialize()
            if not initialization_result:
                # Cache invalidation on error
                instance._invalidate_cache()
                raise RuntimeError("Module initialization failed")
            
            # Cache lifecycle management - Store successful creation
            instance._component_cache[cache_key] = module
            instance._cache_valid = True
            
            return module
            
        except Exception as e:
            # Cache lifecycle management - Invalidation on error
            instance._invalidate_cache()
            
            # Critical errors always logged
            error_msg = f"Failed to create AugmentationUIModule: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    @classmethod
    def create_and_display_augmentation(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul UI Augmentation.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
        """
        from smartcash.ui.core import ui_utils
        
        try:
            module = cls.create_augmentation_module(config=config, **kwargs)
            ui_utils.display_ui_module(
                module=module,
                module_name="Augmentation",
                **kwargs
            )
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"Failed to display Augmentation module: {e}", exc_info=True)
            raise
    
    @classmethod
    def _invalidate_cache(cls):
        """Invalidate the component cache."""
        cls._cache_valid = False
        cls._component_cache.clear()


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
