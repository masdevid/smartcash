"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer dengan error handling yang robust dan fallback minimal
"""

from typing import Dict, Any

def initialize_env_config_ui() -> Dict[str, Any]:
    """Initialize environment config UI dengan error handling"""
    try:
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        
        env_config = EnvConfigComponent()
        env_config.display()
        return env_config.ui_components
        
    except ImportError as e:
        return _create_fallback_ui(f"Missing dependency: {str(e)}")
    except Exception as e:
        return _create_fallback_ui(f"Initialization error: {str(e)}")


def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """Create minimal fallback UI untuk error handling"""
    try:
        from IPython.display import display, HTML
        
        display(HTML(f"""
        <div style="padding: 15px; border: 2px solid #dc3545; border-radius: 8px; 
                   background-color: #fff5f5; margin: 10px 0;">
            <h3>❌ Environment Config Error</h3>
            <p>{error_message}</p>
            <div style="margin-top: 10px; padding: 8px; background-color: #f8f9fa; border-radius: 4px;">
                <strong>Manual Setup:</strong> Mount Drive → Create folders → Copy configs
            </div>
        </div>
        """))
        
        return {'error': error_message, 'manual_setup_required': True}
        
    except Exception:
        print(f"❌ CRITICAL ERROR: {error_message}")
        return {'critical_error': True}


# Alias untuk kompatibilitas
def initialize_environment_config_ui() -> Dict[str, Any]:
    return initialize_env_config_ui()