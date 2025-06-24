# File: ui/setup/env_config/env_config_initializer.py
# Deskripsi: Initializer untuk environment config - orchestration only, tidak ada UI creation

import logging
from typing import Dict, Any, Optional

# Global cache untuk single instance
_ENV_CONFIG_INITIALIZED = False
_CACHED_COMPONENT = None

def initialize_environment_config_ui(force_refresh: bool = False) -> Dict[str, Any]:
    """
    🎯 Initialize environment config UI - ORCHESTRATION ONLY
    
    Flow:
    1. Check cache
    2. Import dan create component via factory
    3. Setup handlers dan state
    4. Return component
    """
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    
    # Return cached component jika sudah initialized
    if _ENV_CONFIG_INITIALIZED and _CACHED_COMPONENT and not force_refresh:
        return _CACHED_COMPONENT
    
    try:
        # Import component class - BUKAN UI creation
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        
        # Create component instance - component yang handle UI creation
        component = EnvConfigComponent()
        
        # Cache component
        _CACHED_COMPONENT = component
        _ENV_CONFIG_INITIALIZED = True
        
        # Return component object langsung
        return component
        
    except Exception as e:
        print(f"🚨 Error initializing environment config: {str(e)}")
        return _create_error_component(str(e))

class ErrorComponent:
    """🚨 Error component dengan display method"""
    def __init__(self, error_msg: str):
        import ipywidgets as widgets
        from IPython.display import display
        
        self.error_msg = error_msg
        self.ui = widgets.HTML(
            value=f"""
            <div style="background: #f8d7da; padding: 15px; border-radius: 6px; 
                        border: 1px solid #f5c6cb; margin: 10px 0;">
                <h4>🚨 Environment Config Error</h4>
                <p><strong>Error:</strong> {error_msg}</p>
                <p><strong>Solusi:</strong> Restart runtime dan coba lagi</p>
            </div>
            """
        )
    
    def display(self):
        from IPython.display import display
        display(self.ui)
    
    def get_ui_components(self):
        return {'ui': self.ui, 'error': self.error_msg}

def _create_error_component(error_msg: str) -> ErrorComponent:
    """🚨 Create error component object"""
    return ErrorComponent(error_msg)

def reset_env_config_ui():
    """🔄 Reset cached component"""
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    
    _ENV_CONFIG_INITIALIZED = False
    _CACHED_COMPONENT = None
    
    print("✅ Environment config cache reset")

# Alias untuk backward compatibility
initialize_env_config_ui = initialize_environment_config_ui

# Export functions
__all__ = [
    'initialize_environment_config_ui',
    'initialize_env_config_ui',
    'reset_env_config_ui'
]