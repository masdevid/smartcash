"""
File: smartcash/ui/dataset/preprocess/__init__.py
Description: Preprocessing module exports with dual API support (UIModule + Legacy)
"""

from typing import Dict, Any, Optional, Union

# ==================== NEW UIMODULE API ====================

from .preprocess_uimodule import (
    PreprocessUIModule,
    create_preprocess_uimodule,
    get_preprocess_uimodule,
    reset_preprocess_uimodule,
    initialize_preprocess_ui_uimodule,
    get_preprocess_components_uimodule
)

# ==================== LEGACY API ====================

from .preprocess_initializer import (
    PreprocessInitializer,
    initialize_preprocess_ui as initialize_preprocess_ui_legacy,
    get_preprocessing_initializer
)

# ==================== DUAL API FUNCTIONS ====================

def initialize_preprocess_ui(
    config: Optional[Dict[str, Any]] = None,
    use_legacy: bool = False,
    display: bool = True,
    **kwargs
) -> Union[PreprocessUIModule, Dict[str, Any], None]:
    """
    Initialize preprocessing UI with dual API support.
    
    Args:
        config: Optional configuration dictionary
        use_legacy: Whether to use legacy initializer (default: False for new UIModule)
        display: Whether to display the UI (requires IPython)
        **kwargs: Additional arguments
        
    Returns:
        PreprocessUIModule instance (new API) or components dict (legacy API)
    """
    if use_legacy:
        # Use legacy initializer API
        if display:
            initialize_preprocess_ui_legacy(config=config, **kwargs)
            return None
        else:
            # Return components without display
            initializer = get_preprocessing_initializer()
            return initializer.initialize(config=config, **kwargs)
    else:
        # Use new UIModule API
        return initialize_preprocess_ui_uimodule(
            config=config,
            display=display,
            **kwargs
        )


def display_preprocess_ui(
    config: Optional[Dict[str, Any]] = None,
    use_legacy: bool = False,
    **kwargs
) -> None:
    """
    Display preprocessing UI with dual API support.
    
    Args:
        config: Optional configuration dictionary
        use_legacy: Whether to use legacy initializer
        **kwargs: Additional arguments
    """
    initialize_preprocess_ui(
        config=config,
        use_legacy=use_legacy,
        display=True,
        **kwargs
    )


def get_preprocess_components(use_legacy: bool = False) -> Dict[str, Any]:
    """
    Get preprocessing UI components with dual API support.
    
    Args:
        use_legacy: Whether to use legacy initializer
        
    Returns:
        Dictionary of UI components
    """
    if use_legacy:
        # Use legacy API
        initializer = get_preprocessing_initializer()
        return initializer.get_components() if hasattr(initializer, 'get_components') else {}
    else:
        # Use new UIModule API
        return get_preprocess_components_uimodule()


# ==================== EXPORTS ====================

__all__ = [
    # NEW UIMODULE API
    'PreprocessUIModule',
    'create_preprocess_uimodule',
    'get_preprocess_uimodule', 
    'reset_preprocess_uimodule',
    'initialize_preprocess_ui_uimodule',
    'get_preprocess_components_uimodule',
    
    # LEGACY API
    'PreprocessInitializer',
    'initialize_preprocess_ui_legacy',
    'get_preprocessing_initializer',
    
    # DUAL API (RECOMMENDED)
    'initialize_preprocess_ui',
    'display_preprocess_ui',
    'get_preprocess_components'
]