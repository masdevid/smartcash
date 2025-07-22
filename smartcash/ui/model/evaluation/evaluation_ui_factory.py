"""
Factory untuk membuat dan menampilkan modul UI Evaluation.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Evaluation menggunakan BaseUIModule dan UI Factory pattern.

file_path: /Users/masdevid/Projects/smartcash/smartcash/ui/model/evaluation/evaluation_ui_factory.py
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
from smartcash.ui.logger import get_module_logger

class EvaluationUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Evaluation.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Evaluation dengan konfigurasi default yang sesuai.
    
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
    def create_evaluation_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> EvaluationUIModule:
        """
        Buat instance EvaluationUIModule dengan cache lifecycle management.
        
        Args:
            config: Konfigurasi opsional untuk modul
            force_refresh: Force refresh cache if True
            **kwargs: Argumen tambahan untuk inisialisasi modul
                
        Returns:
            Instance EvaluationUIModule yang sudah diinisialisasi dengan caching
        """
        logger = get_module_logger(__name__)
        instance = cls()
        
        try:
            # Cache lifecycle management - Creation phase
            cache_key = f"evaluation_module_{hash(str(config))}"
            
            # Check cache validity
            if not force_refresh and instance._cache_valid and cache_key in instance._component_cache:
                cached_module = instance._component_cache[cache_key]
                if cached_module and hasattr(cached_module, '_is_initialized') and cached_module._is_initialized:
                    # Cache hit - return cached instance
                    if hasattr(cached_module, 'log_debug'):
                        cached_module.log_debug("Menggunakan cached instance EvaluationUIModule")
                    return cached_module
            
            # Cache miss or invalid - create new instance
            module = EvaluationUIModule()
            
            # Use module's logging when available
            if hasattr(module, 'log_debug'):
                module.log_debug("Membuat instance EvaluationUIModule baru")
            
            # Initialize the module with config if provided
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
            
            # Log success
            if hasattr(module, 'log_debug'):
                module.log_debug("✅ Berhasil menginisialisasi EvaluationUIModule")
                
            return module
            
        except Exception as e:
            # Cache lifecycle management - Invalidation on error
            if 'instance' in locals():
                instance._invalidate_cache()
            
            # Log error with module logger if available, fallback to factory logger
            error_msg = f"Gagal membuat EvaluationUIModule: {e}"
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise
    
    @classmethod
    def _invalidate_cache(cls):
        """Invalidate the component cache."""
        cls._cache_valid = False
        cls._component_cache.clear()
    
    @classmethod
    def create_and_display_evaluation(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Buat dan tampilkan modul Evaluation UI.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Argumen tambahan untuk inisialisasi modul
                - auto_display: Boolean, apakah akan menampilkan UI secara otomatis (default: True)
            
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        logger = get_module_logger(__name__)
        module = None

        try:
            # Buat instance modul
            module = cls.create_evaluation_module(config=config, **kwargs)
            
            # Tampilkan UI menggunakan utility yang konsisten
            ui_utils.display_ui_module(
                module=module,
                module_name="Evaluation",
                **kwargs
            )

            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Evaluation UI: {str(e)}"
            # Try to log to module first, fallback to factory logger
            if module is not None and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


def create_evaluation_display(**kwargs) -> callable:
    """
    Create a display function for the evaluation UI.
    
    This is a convenience function that returns a callable that can be used
    to display the evaluation UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the evaluation UI
        
    Returns:
        A callable that will display the evaluation UI when called
    """
    def display_fn():
        """Display the evaluation UI with the configured settings."""
        EvaluationUIFactory.create_and_display_evaluation(**kwargs)
        return None
    
    return display_fn
