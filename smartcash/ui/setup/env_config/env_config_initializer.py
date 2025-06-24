"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Initializer dengan proper error handling dan fallback yang minimal
"""

from typing import Dict, Any

def initialize_env_config_ui() -> Dict[str, Any]:
    """🚀 Initialize environment config UI dengan error handling yang proper"""
    try:
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        from smartcash.ui.utils.logger_bridge import setup_ui_logging
        
        # Create component
        env_config = EnvConfigComponent()
        
        # Ensure log output terdaftar
        if 'log_output' not in env_config.ui_components and 'log_accordion' in env_config.ui_components:
            env_config.ui_components['log_output'] = env_config.ui_components['log_accordion'].children[0]
        
        # Setup logging dengan UI components
        setup_ui_logging(env_config.ui_components, "env_config")
        
        # Tampilkan UI
        env_config.display()
        
        return env_config.ui_components
        
    except Exception as e:
        # 🔧 Fallback sederhana dengan proper error info
        error_msg = str(e)
        
        print(f"❌ Environment Config Error: {error_msg}")
        print("💡 Manual setup diperlukan:")
        print("   1. Mount Google Drive")
        print("   2. Buat folder: /content/drive/MyDrive/SmartCash")
        print("   3. Copy configs dari /content/smartcash/configs")
        
        # Return structured error response
        return {
            'error': error_msg,
            'manual_setup_required': True,
            'status': 'failed',
            'status_message': f'Initialization failed: {error_msg}',
            'troubleshooting': {
                'drive_mount': 'Mount Google Drive terlebih dahulu',
                'config_copy': 'Copy manual configs dari repo ke Drive',
                'folder_creation': 'Buat folder SmartCash di Drive'
            }
        }

# Alias untuk backward compatibility
def initialize_environment_config_ui() -> Dict[str, Any]:
    """🔄 Alias untuk kompatibilitas dengan existing code"""
    return initialize_env_config_ui()