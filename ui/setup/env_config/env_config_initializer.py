"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Inisialisasi UI untuk konfigurasi environment - diperbaiki dengan error handling yang robust
"""

from typing import Dict, Any

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi konfigurasi environment dengan error handling yang robust
    
    Returns:
        Dictionary UI components
    """
    try:
        # Import component
        from smartcash.ui.setup.env_config.components.env_config_component import EnvConfigComponent
        
        # Buat dan tampilkan component
        env_config = EnvConfigComponent()
        env_config.display()
        
        # Return ui_components untuk kompatibilitas
        return env_config.ui_components
        
    except ImportError as e:
        # Handle missing dependencies
        return _create_fallback_ui(f"Missing dependency: {str(e)}")
        
    except Exception as e:
        # Handle general errors
        return _create_fallback_ui(f"Error initializing environment config: {str(e)}")


def _create_fallback_ui(error_message: str) -> Dict[str, Any]:
    """
    Buat UI fallback minimal untuk error handling
    
    Args:
        error_message: Pesan error yang terjadi
        
    Returns:
        Dictionary UI components fallback
    """
    try:
        from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
        from IPython.display import display
        
        # Create error UI dengan factory
        error_components = UIFactory.create_error_ui_components(error_message)
        
        # Display error UI
        display(error_components['ui_layout'])
        
        return error_components
        
    except Exception as fallback_error:
        # Ultimate fallback - pure text output
        try:
            from IPython.display import display, HTML
            
            fallback_html = f"""
            <div style="padding: 20px; border: 2px solid #dc3545; border-radius: 8px; 
                       background-color: #fff5f5; color: #721c24; margin: 10px 0;">
                <h3>❌ Error Setup Environment Config</h3>
                <p><strong>Primary Error:</strong> {error_message}</p>
                <p><strong>Fallback Error:</strong> {str(fallback_error)}</p>
                <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                    <h4>🔧 Manual Setup Required</h4>
                    <p>Silakan setup environment secara manual:</p>
                    <ol>
                        <li>Pastikan Google Drive ter-mount (jika di Colab)</li>
                        <li>Buat direktori: data, exports, logs, models, output</li>
                        <li>Copy file config dari smartcash/configs/</li>
                    </ol>
                </div>
            </div>
            """
            
            display(HTML(fallback_html))
            
            return {
                'error': error_message,
                'fallback_error': str(fallback_error),
                'manual_setup_required': True
            }
            
        except Exception:
            # Absolute last resort - print to console
            print(f"❌ CRITICAL ERROR: Unable to initialize environment config UI")
            print(f"Primary Error: {error_message}")
            print(f"Fallback Error: {str(fallback_error)}")
            print("Please setup environment manually.")
            
            return {
                'critical_error': True,
                'error': error_message,
                'fallback_error': str(fallback_error)
            }


# Alias untuk kompatibilitas mundur
def initialize_environment_config_ui() -> Dict[str, Any]:
    """Alias untuk kompatibilitas mundur"""
    return initialize_env_config_ui()