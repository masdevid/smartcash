"""
Optimized factory untuk membuat dan menampilkan modul UI Visualization.

Streamlined factory yang menyediakan pembuatan dan tampilan modul visualization
dengan penanganan data kosong yang robust dan performa yang dioptimasi.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.dataset.visualization.visualization_uimodule import VisualizationUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function
from smartcash.common.logger import get_module_logger

class VisualizationUIFactory(UIFactory):
    """
    Optimized factory untuk modul UI Visualization dengan placeholder support.
    
    Menyediakan pembuatan modul yang robust dengan fallback untuk data kosong
    dan error handling yang lebih baik.
    
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
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> VisualizationUIModule:
        """
        Create a new instance of VisualizationUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New VisualizationUIModule instance
        """
        module = VisualizationUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
    @classmethod
    def create_visualization_module(
        cls,
        config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> VisualizationUIModule:
        """
        Create optimized VisualizationUIModule instance with robust error handling and caching.
        
        Args:
            config: Optional configuration for the module
            force_refresh: Force refresh cache if True
            **kwargs: Additional initialization arguments
                
        Returns:
            Fully initialized VisualizationUIModule with placeholder support and caching
        """
        return create_ui_factory_method(
            module_class=VisualizationUIModule,
            module_name="Visualization",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
    @classmethod
    def _ensure_placeholder_cards(cls, module: VisualizationUIModule) -> None:
        """
        Ensure visualization stats cards show placeholder data when backend fails.
        
        Args:
            module: The initialized visualization module
        """
        try:
            # Create empty placeholder stats for cards
            empty_stats = {
                'success': True,  # Mark as successful to show cards with 0 values
                'dataset_stats': {
                    'overview': {'total_files': 0},
                    'by_split': {
                        'train': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                        'valid': {'raw': 0, 'preprocessed': 0, 'augmented': 0},
                        'test': {'raw': 0, 'preprocessed': 0, 'augmented': 0}
                    }
                },
                'augmentation_stats': {
                    'by_split': {
                        'train': {'file_count': 0},
                        'valid': {'file_count': 0},
                        'test': {'file_count': 0}
                    }
                },
                'data_directory': 'No data directory configured',
                'last_updated': 'Placeholder - Click refresh to load real data'
            }
            
            # Update dashboard cards with placeholder data
            if hasattr(module, '_dashboard_cards') and module._dashboard_cards:
                module._dashboard_cards.update_all_cards(empty_stats)
                
            # Store placeholder stats
            if hasattr(module, '_latest_stats'):
                module._latest_stats = empty_stats
                
        except Exception as e:
            # Silently continue if placeholder setup fails - better than crashing
            if hasattr(module, 'log_warning'):
                module.log_warning(f"Failed to setup placeholder cards: {e}")
    
    @classmethod
    def create_and_display_visualization(
        cls,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Create and display optimized Visualization UI with placeholders.
        
        Args:
            config: Optional module configuration
            **kwargs: Additional display arguments
                - auto_display: Boolean, whether to automatically display the UI (default: True)
                - refresh_cache: Boolean, whether to force refresh the cache (default: False)
                
        Returns:
            None (displays the UI using IPython.display)
        """
        from smartcash.ui.core import ui_utils
        logger = get_module_logger(__name__)
        
        try:
            # Get force_refresh from kwargs if provided
            force_refresh = kwargs.pop('force_refresh', False)
            
            # Create module with optimizations and cache management
            module = cls.create_visualization_module(
                config=config, 
                force_refresh=force_refresh,
                **kwargs
            )
            
            # Display UI using consistent utility
            ui_utils.display_ui_module(
                module=module,
                module_name="Visualization",
                **kwargs
            )
            
            # Return None explicitly to avoid displaying module object
            return None
            
        except Exception as e:
            error_msg = f"Gagal membuat dan menampilkan Visualization UI: {str(e)}"
            # Try to log to module first, fallback to factory logger
            if 'module' in locals() and hasattr(module, 'log_error'):
                module.log_error(error_msg)
            else:
                logger.error(error_msg, exc_info=True)
            raise


def create_visualization_display(**kwargs) -> callable:
    """
    Create a display function for the visualization UI.
    
    This is a convenience function that returns a callable that can be used
    to display the visualization UI with the given configuration.
    
    Args:
        **kwargs: Configuration options for the visualization UI
        
    Returns:
        A callable that will display the visualization UI when called
    """
    def display_fn():
        VisualizationUIFactory.create_and_display_visualization(**kwargs)
        return None
    
    return display_fn
