"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Environment config initializer dengan auto-display untuk Colab
"""

from typing import Dict, Any, Optional
from IPython.display import display

# Global state untuk caching
_ENV_CONFIG_INITIALIZED = False
_CACHED_COMPONENT = None

def initialize_env_config_ui(force_refresh: bool = False, auto_display: bool = True):
    """
    🚀 Initialize environment config UI dengan caching dan auto-display
    
    Args:
        force_refresh: Force refresh UI components
        auto_display: Auto display UI (True untuk direct call, False untuk assignment)
        
    Returns:
        Component instance untuk chaining (dengan auto-display)
    """
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    
    # Return cached component jika sudah initialized
    if _ENV_CONFIG_INITIALIZED and _CACHED_COMPONENT and not force_refresh:
        if auto_display:
            _CACHED_COMPONENT.display()
        return _CACHED_COMPONENT
    
    try:
        # Import with cache diagnosis
        try:
            from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        except ImportError as e:
            print(f"🚨 Import Error: {e}")
            print("🔄 Trying cache clear...")
            _clear_import_cache()
            from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        
        # Create component
        component = EnvConfigComponent()
        
        # Auto-display for direct calls
        if auto_display:
            component.display()
        
        # Cache component
        _CACHED_COMPONENT = component
        _ENV_CONFIG_INITIALIZED = True
        
        return component
        
    except Exception as e:
        print(f"🚨 Error initializing env config UI: {e}")
        
        # Create fallback with auto-display
        fallback_component = _create_fallback_component(str(e))
        if auto_display:
            display(fallback_component)
        
        return fallback_component

def initialize_environment_config_ui(force_refresh: bool = False, auto_display: bool = True):
    """🔄 Alias untuk backward compatibility"""
    return initialize_env_config_ui(force_refresh=force_refresh, auto_display=auto_display)

def _clear_import_cache():
    """🧹 Clear import cache untuk module yang bermasalah"""
    import sys
    modules_to_clear = []
    
    # Find modules to clear
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('smartcash.ui.setup.env_config'):
            modules_to_clear.append(module_name)
    
    # Clear modules
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print(f"🧹 Cleared {len(modules_to_clear)} cached modules")

def _create_fallback_component(error_msg: str):
    """🚨 Create fallback component untuk error handling"""
    import ipywidgets as widgets
    
    return widgets.HTML(
        value=f"""
        <div style="background: #f8d7da; padding: 20px; border-radius: 8px; border: 1px solid #f5c6cb; margin: 10px 0px;">
            <h3>🚨 Error Loading Environment Config</h3>
            <p><strong>Error:</strong> {error_msg}</p>
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h4>🔧 Solutions:</h4>
                <ol>
                    <li><strong>Restart Runtime:</strong> Runtime → Restart runtime</li>
                    <li><strong>Clear Cache:</strong> Run <code>reset_env_config_ui()</code></li>
                    <li><strong>Force Refresh:</strong> Run <code>initialize_env_config_ui(force_refresh=True)</code></li>
                </ol>
            </div>
        </div>
        """
    )

def reset_env_config_ui():
    """🔄 Reset cached UI dan clear import cache"""
    global _ENV_CONFIG_INITIALIZED, _CACHED_COMPONENT
    
    # Clear global cache
    _ENV_CONFIG_INITIALIZED = False
    _CACHED_COMPONENT = None
    
    # Clear import cache
    _clear_import_cache()
    
    print("✅ Environment config UI cache reset")
    print("🔄 Re-run initialize_env_config_ui() to reload")

# Monkey patch untuk IPython auto-display
def _ipython_display_(self):
    """Enable auto-display dalam Jupyter/Colab cells"""
    return self.display()

# Export functions
__all__ = [
    'initialize_env_config_ui',
    'initialize_environment_config_ui', 
    'reset_env_config_ui'
]