"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer sederhana untuk environment config UI tanpa fallback berlebihan
"""

from typing import Dict, Any

def initialize_env_config_ui() -> Dict[str, Any]:
    """Initialize environment config UI dengan approach sederhana"""
    try:
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        from smartcash.ui.utils.logger_bridge import setup_ui_logging
        
        # Create component
        env_config = EnvConfigComponent()
        
        # Pastikan log output terdaftar
        if 'log_output' not in env_config.ui_components and 'log_accordion' in env_config.ui_components:
            env_config.ui_components['log_output'] = env_config.ui_components['log_accordion'].children[0]
        
        # Setup logging dengan UI components
        setup_ui_logging(env_config.ui_components, "env_config")
        
        # Tampilkan UI
        env_config.display()
        
        return env_config.ui_components
        
    except Exception as e:
        # Simple fallback tanpa UI kompleks
        print(f"❌ Environment Config Error: {str(e)}")
        print("💡 Manual setup required: Mount Drive → Create folders → Copy configs")
        
        return {
            'error': str(e), 
            'manual_setup_required': True,
            'status': 'failed'
        }

# Alias untuk kompatibilitas
def initialize_environment_config_ui() -> Dict[str, Any]:
    """Alias untuk backward compatibility"""
    return initialize_env_config_ui()