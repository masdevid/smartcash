"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Environment config initializer dengan return value tanpa assignment
"""

from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent

# Global state untuk caching
_ENV_CONFIG_INITIALIZED = False
_CACHED_COMPONENT = None

def initialize_env_config_ui(force_refresh: bool = False) -> EnvConfigComponent:
    """
    🚀 Initialize environment config UI dengan caching
    
    Args:
        force_refresh: Force refresh UI components
        
    Returns:
        EnvConfigComponent: Component instance untuk chaining
    """
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    
    # Return cached component jika sudah initialized
    if _ENV_CONFIG_INITIALIZED and _CACHED_COMPONENT and not force_refresh:
        return _CACHED_COMPONENT
    
    try:
        # Create component
        component = EnvConfigComponent()
        
        # Display UI
        component.display()
        
        # Cache component
        _CACHED_COMPONENT = component
        _ENV_CONFIG_INITIALIZED = True
        
        return component
        
    except Exception as e:
        print(f"🚨 Error initializing env config UI: {e}")
        
        # Create fallback component
        fallback_component = _create_fallback_component(str(e))
        display(fallback_component)
        
        return fallback_component

def initialize_environment_config_ui(force_refresh: bool = False) -> EnvConfigComponent:
    """🔄 Alias untuk backward compatibility"""
    return initialize_env_config_ui(force_refresh=force_refresh)

def _create_fallback_component(error_msg: str):
    """🚨 Create fallback component untuk error handling"""
    import ipywidgets as widgets
    
    return widgets.HTML(
        value=f"""
        <div style="background: #f8d7da; padding: 20px; border-radius: 8px; border: 1px solid #f5c6cb; margin: 10px 0px;">
            <h3>🚨 Error Loading Environment Config</h3>
            <p><strong>Error:</strong> {error_msg}</p>
            <p>Silakan refresh cell atau restart runtime untuk mencoba lagi.</p>
        </div>
        """
    )

# Reset function untuk development
def reset_env_config_ui():
    """🔄 Reset cached UI untuk development"""
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    _ENV_CONFIG_INITIALIZED = False
    _CACHED_COMPONENT = None
    print("✅ Environment config UI cache reset")