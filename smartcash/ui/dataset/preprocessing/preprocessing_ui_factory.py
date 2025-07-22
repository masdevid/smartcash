"""
Factory untuk membuat dan menampilkan modul UI Preprocessing.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Preprocessing menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule
from smartcash.ui.logger import get_module_logger

class PreprocessingUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Preprocessing.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Preprocessing dengan konfigurasi default yang sesuai.
    
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
                
        Returns:
            Instance PreprocessingUIModule yang sudah diinisialisasi dengan caching
        """
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            # Cache lifecycle management - Creation phase
            cache_key = f"preprocessing_module_{hash(str(config))}"
            
            # Check cache validity
            if not force_refresh and instance._cache_valid and cache_key in instance._component_cache:
                cached_module = instance._component_cache[cache_key]
                if cached_module and hasattr(cached_module, '_is_initialized') and cached_module._is_initialized:
                    # Cache hit - return cached instance
                    return cached_module
            
            # Cache miss or invalid - create new instance
            enable_environment = kwargs.get('enable_environment', False)
            module = PreprocessingUIModule(enable_environment=enable_environment)
            
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
            error_msg = f"Failed to create PreprocessingUIModule: {e}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def _invalidate_cache(self) -> None:
        """Cache lifecycle management - Invalidation phase."""
        self._cache_valid = False
        
        # Widget lifecycle - Proper cleanup of cached components
        for module in self._component_cache.values():
            try:
                if hasattr(module, 'cleanup'):
                    module.cleanup()
                # Additional IPython widget cleanup
                if hasattr(module, '_ui_components') and module._ui_components:
                    for component in module._ui_components.values():
                        if hasattr(component, 'close'):
                            component.close()
            except Exception:
                pass  # Ignore cleanup errors
        
        # Cache cleanup
        self._component_cache.clear()
    
    @classmethod
    def clear_cache(cls) -> None:
        """Cache lifecycle management - Manual cleanup."""
        instance = cls()
        instance._invalidate_cache()
    
    @classmethod
    def create_and_display_preprocessing(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Preprocessing UI dengan optimized performance.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
                - force_refresh: Force cache refresh (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        logger = get_module_logger(__name__)

        try:
            # Lazy loading - create module only when needed
            module = cls.create_preprocessing_module(config=config, **kwargs)
            
            # Memory management - display with minimal resource usage
            ui_utils.display_ui_module(
                module=module,
                module_name="Preprocessing",
                **kwargs
            )

            # Return None explicitly to avoid displaying module object
            return None
            
        except Exception as e:
            # Critical errors always logged
            error_msg = f"Failed to create and display Preprocessing UI: {str(e)}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


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
