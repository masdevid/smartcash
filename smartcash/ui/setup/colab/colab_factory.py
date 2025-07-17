"""
File: smartcash/ui/setup/colab/colab_factory.py
Description: Factory functions for creating and initializing Colab UI modules.
"""

from typing import Dict, Any, Optional
from smartcash.ui.logger import get_module_logger


def initialize_colab_ui(config: Optional[Dict[str, Any]] = None, 
                       show_display: bool = True, 
                       **kwargs) -> Optional[Dict[str, Any]]:
    """Initialize and optionally display the Colab UI module."""
    from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory
    from .colab_uimodule import ColabUIModule
    
    # Filter out conflicting display-related parameters from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['display', 'show_display']}
    
    # Determine final display value - prioritize explicit 'display' parameter
    if 'display' in kwargs:
        final_display = kwargs['display']
    else:
        final_display = show_display
    
    return EnhancedUIModuleFactory.create_and_display(
        module_class=ColabUIModule,
        config=config,
        display=final_display,
        **filtered_kwargs
    )


def get_colab_components(config: Optional[Dict[str, Any]] = None, 
                        **kwargs) -> Optional[Dict[str, Any]]:
    """Get Colab UI components without displaying."""
    return initialize_colab_ui(config=config, show_display=False, **kwargs)


def create_colab_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> 'ColabUIModule':
    """
    Create a new Colab UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        ColabUIModule instance
    """
    from .colab_uimodule import ColabUIModule
    
    module = ColabUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def display_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Display Colab UI and return components."""
    return initialize_colab_ui(config=config, display=True, **kwargs)


def detect_colab_environment() -> Dict[str, Any]:
    """Detect if running in Google Colab environment."""
    try:
        import google.colab  # noqa: F401
        return {"is_colab": True, "runtime_type": "colab"}
    except ImportError:
        return {"is_colab": False, "runtime_type": "local"}


def mount_google_drive(drive_path: str = "/content/drive") -> Dict[str, Any]:
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount(drive_path)
        return {"success": True, "path": drive_path}
    except ImportError:
        return {"success": False, "error": "Not running in Google Colab"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_colab_shared_methods() -> None:
    """Register shared methods for Colab module."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register Colab-specific shared methods
        SharedMethodRegistry.register_method(
            'colab.detect_environment',
            detect_colab_environment,
            description='Detect Colab environment'
        )
        
        SharedMethodRegistry.register_method(
            'colab.mount_drive',
            mount_google_drive,
            description='Mount Google Drive'
        )
        
        SharedMethodRegistry.register_method(
            'colab.get_status',
            lambda: create_colab_uimodule().get_colab_status(),
            description='Get Colab environment status'
        )
        
        logger = get_module_logger("smartcash.ui.setup.colab.factory")
        logger.debug("📋 Registered Colab shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.setup.colab.factory")
        logger.error(f"Failed to register shared methods: {e}")


# Global module instance for singleton pattern
_colab_module_instance: Optional['ColabUIModule'] = None


def get_colab_uimodule() -> Optional['ColabUIModule']:
    """Get the current Colab UIModule instance."""
    global _colab_module_instance
    return _colab_module_instance


def reset_colab_uimodule() -> None:
    """Reset the global Colab UIModule instance."""
    global _colab_module_instance
    if _colab_module_instance:
        try:
            _colab_module_instance.cleanup()
        except:
            pass
    _colab_module_instance = None


# Auto-register when module is imported
try:
    register_colab_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")