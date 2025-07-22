"""
Factory untuk membuat dan menampilkan modul UI Backbone.

File ini menyediakan factory khusus untuk membuat dan menampilkan
modul UI Backbone menggunakan BaseUIModule dan UI Factory pattern.
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.ui_factory import UIFactory
from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule
from smartcash.ui.core.utils import create_ui_factory_method, create_display_function

class BackboneUIFactory(UIFactory):
    """
    Factory untuk membuat dan menampilkan modul UI Backbone.
    
    Kelas ini menyediakan method khusus untuk membuat dan menampilkan
    modul UI Backbone dengan konfigurasi default yang sesuai.
    
    Features (compliant with optimization.md):
    - 🚀 Leverages parent's cache lifecycle management for component reuse
    - 💾 Lazy loading of UI components
    - 🧹 Proper widget lifecycle cleanup
    - 📝 Minimal logging for performance
    """
    
    @classmethod
    def _create_module_instance(cls, config: Optional[Dict[str, Any]] = None, **kwargs) -> BackboneUIModule:
        """
        Create a new instance of BackboneUIModule.
        
        Args:
            config: Konfigurasi opsional untuk modul
            **kwargs: Additional arguments for module initialization
                
        Returns:
            New BackboneUIModule instance
        """
        module = BackboneUIModule()
        
        # Apply config if provided
        if config is not None and hasattr(module, 'update_config'):
            module.update_config(config)
            
        return module
    
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
        return create_ui_factory_method(
            module_class=BackboneUIModule,
            module_name="Backbone",
            create_module_func=cls._create_module_instance
        )(config=config, force_refresh=force_refresh, **kwargs)
    
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
        display_fn = create_display_function(
            factory_class=cls,
            create_method_name='create_backbone_module',
            module_name='Backbone',
            config=config,
            **kwargs
        )
        return display_fn()


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
