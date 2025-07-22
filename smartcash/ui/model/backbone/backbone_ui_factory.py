"""
Factory untuk membuat dan menampilkan modul UI Backbone.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Backbone menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
from smartcash.ui.logger import get_module_logger

class BackboneUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Backbone.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Backbone dengan konfigurasi default yang sesuai.
    
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
    def create_backbone_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> BackboneUIModule:
        """
        Buat instance BackboneUIModule dengan konfigurasi yang diberikan.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Paksa pembaruan cache jika True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance BackboneUIModule yang sudah diinisialisasi dengan caching
        """
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            # Cache lifecycle management - Creation phase
            cache_key = f"backbone_module_{hash(str(config))}"
            
            # Check cache validity
            if not force_refresh and instance._cache_valid and cache_key in instance._component_cache:
                cached_module = instance._component_cache[cache_key]
                if cached_module and hasattr(cached_module, '_is_initialized') and cached_module._is_initialized:
                    # Cache hit - return cached instance
                    if hasattr(cached_module, 'log_debug'):
                        cached_module.log_debug("✅ Menggunakan instance BackboneUIModule dari cache")
                    else:
                        logger.debug("✅ Menggunakan instance BackboneUIModule dari cache")
                    return cached_module
            
            # Cache miss or invalid - create new instance
            module = BackboneUIModule()
            
            # Apply config if provided
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
            if 'instance' in locals():
                instance._invalidate_cache()
                
            # Critical errors always logged
            error_msg = f"Gagal membuat BackboneUIModule: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
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
                - force_refresh: Boolean, apakah akan memaksa refresh cache (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        logger = get_module_logger(__name__)
        
        try:
            # Get force_refresh from kwargs if provided
            force_refresh = kwargs.pop('force_refresh', False)
            
            # Buat instance modul dengan manajemen cache
            module = cls.create_backbone_module(
                config=config,
                force_refresh=force_refresh,
                **kwargs
            )
            
            # Tampilkan UI menggunakan utility yang konsisten
            ui_utils.display_ui_module(
                module=module,
                module_name="Backbone",
                **kwargs
            )
            
            # Return None explicitly to avoid displaying module object
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Backbone UI: {str(e)}"
            # Try to log to module first, fallback to factory logger
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


def create_backbone_display(**kwargs) -> callable:
    """
    Create a display function for the backbone UI.
    
    This is a convenience function that returns a callable that can be used
    to display the backbone UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the backbone UI
            - config: Optional configuration dictionary for the module
            - force_refresh: Boolean, whether to force refresh the cache (default: False)
            - auto_display: Boolean, whether to automatically display the UI (default: True)
        
    Returns:
        A callable that will display the backbone UI when called
        
    Example:
        ```python
        # Create and display the backbone UI
        display_fn = create_backbone_display(config=my_config)
        display_fn()  # This will display the UI
        ```
    """
    def display_fn():
        try:
            BackboneUIFactory.create_and_display_backbone(**kwargs)
            return None
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"Failed to display backbone UI: {e}", exc_info=True)
            raise
    
    return display_fn
