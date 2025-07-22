"""
Factory untuk membuat dan menampilkan modul UI Split.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Split menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.split.split_uimodule import SplitUIModule
from smartcash.ui.logger import get_module_logger

class SplitUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Split.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Split dengan konfigurasi default yang sesuai.
    
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
    
    def _invalidate_cache(self):
        """Invalidate the component cache and mark as invalid."""
        self._component_cache.clear()
        self._cache_valid = False
        if hasattr(self, '_instance'):
            self._instance = None
    
    @classmethod
    def reset_cache(cls):
        """Explicitly clear the factory singleton cache and invalidate all components."""
        instance = cls()
        instance._invalidate_cache()
        instance._initialized = False
    
    @classmethod
    def create_split_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> SplitUIModule:
        """
        Buat instance SplitUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance SplitUIModule yang sudah diinisialisasi dengan caching
        """
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            # Cache lifecycle management - Creation phase
            cache_key = f"split_module_{hash(str(config))}"
            
            # Check cache validity
            if not force_refresh and instance._cache_valid and cache_key in instance._component_cache:
                cached_module = instance._component_cache[cache_key]
                if cached_module and hasattr(cached_module, '_is_initialized') and cached_module._is_initialized:
                    # Cache hit - return cached instance
                    return cached_module
            
            # Cache miss or invalid - create new instance
            module = SplitUIModule()
            
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
            error_msg = f"Gagal membuat SplitUIModule: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise
    
    @classmethod
    def create_and_display_split(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Split UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        logger = get_module_logger(__name__)

        try:
            # Buat instance modul
            module = cls.create_split_module(config=config, **kwargs)
            
            # Tampilkan UI menggunakan utility yang konsisten
            ui_utils.display_ui_module(

                module=module,

                module_name="Split",

                **kwargs

            )

            # Return None explicitly to avoid displaying module object

            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Split UI: {str(e)}"
            # Try to log to module first, fallback to factory logger
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


def create_split_display(**kwargs) -> callable:
    """
    Create a display function for the split UI.
    
    This is a convenience function that returns a callable that can be used
    to display the split UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the split UI
        
    Returns:
        A callable that will display the split UI when called
    """
    def display_fn():
        SplitUIFactory.create_and_display_split(**kwargs)
        return None
    
    return display_fn
