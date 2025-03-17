"""
File: smartcash/ui/setup/env_config_fallbacks.py
Deskripsi: Fallback handlers untuk konfigurasi environment SmartCash
"""

from typing import Dict, Any

def handle_environment_detection_error(
    ui_components: Dict[str, Any], 
    error: Exception
) -> Dict[str, Any]:
    """
    Handler untuk error deteksi environment.
    
    Args:
        ui_components: Dictionary komponen UI
        error: Exception yang terjadi
        
    Returns:
        Dictionary UI components yang telah diperbarui
    """
    # Tambahkan pesan error ke status output
    if 'status' in ui_components:
        with ui_components['status']:
            from IPython.display import display, HTML
            display(HTML(f"""
            <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px;">
                ❌ Kesalahan Deteksi Environment: {str(error)}
                <p>Silakan periksa konfigurasi sistem anda dan pastikan semua dependensi terpasang.</p>
            </div>
            """))
    
    return ui_components

def create_minimal_environment_config() -> Dict[str, Any]:
    """
    Buat konfigurasi environment minimal sebagai fallback.
    
    Returns:
        Dictionary konfigurasi environment minimal
    """
    return {
        'environment_type': 'unknown',
        'base_dir': '.',
        'is_colab': False,
        'is_notebook': False
    }