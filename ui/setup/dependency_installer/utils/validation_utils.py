"""
File: smartcash/ui/setup/dependency_installer/utils/validation_utils.py
Deskripsi: Utilitas untuk validasi komponen UI dependency installer
"""

from typing import Dict, Any, List, Tuple

def validate_ui_components(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi komponen UI dependency installer
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary dengan hasil validasi
    """
    # Komponen kritis yang harus ada
    critical_components = [
        'main_container', 
        'install_button', 
        'log_output', 
        'progress_container'
    ]
    
    # Validasi keberadaan komponen kritis
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    
    if missing_components:
        return {
            'valid': False,
            'message': f"Komponen UI tidak lengkap: {', '.join(missing_components)}"
        }
    
    # Validasi fungsi tombol install
    if not hasattr(ui_components['install_button'], 'on_click'):
        return {
            'valid': False,
            'message': "Install button tidak memiliki handler on_click"
        }
    
    # Validasi handler tombol install
    if not ui_components['install_button']._click_handlers.callbacks:
        return {
            'valid': False,
            'message': "Install button tidak memiliki callback"
        }
    
    # Semua validasi berhasil
    return {
        'valid': True,
        'message': "Komponen UI valid"
    }

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi konfigurasi dependency installer
    
    Args:
        config: Dictionary konfigurasi
        
    Returns:
        Dictionary dengan hasil validasi
    """
    # Konfigurasi kritis yang harus ada
    critical_config = [
        'auto_install',
        'selected_packages'
    ]
    
    # Validasi keberadaan konfigurasi kritis
    missing_config = [conf for conf in critical_config if conf not in config]
    
    if missing_config:
        return {
            'valid': False,
            'message': f"Konfigurasi tidak lengkap: {', '.join(missing_config)}"
        }
    
    # Semua validasi berhasil
    return {
        'valid': True,
        'message': "Konfigurasi valid"
    }

def get_default_config() -> Dict[str, Any]:
    """Mendapatkan konfigurasi default untuk dependency installer
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        'auto_install': False,
        'selected_packages': ['yolov5_req', 'smartcash_req', 'torch_req'],
        'custom_packages': '',
        'validate_after_install': True,
        'delay_analysis': True,  # Flag untuk menunda analisis sampai UI terender
        'suppress_logs': True,   # Tekan log selama inisialisasi
        'hide_progress': True,   # Sembunyikan progress selama inisialisasi
    }
