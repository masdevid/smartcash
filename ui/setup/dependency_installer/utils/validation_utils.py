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
        Dictionary dengan hasil validasi atau ui_components jika valid
    """
    import logging
    from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
    
    # Setup logger dengan namespace yang konsisten
    MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, "DEPS")
    logger = logging.getLogger(DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    logger.info("Validating UI components")
    
    # Jika ada error di ui_components, kembalikan tanpa validasi
    if 'error' in ui_components:
        logger.error(f"Skipping validation due to error: {ui_components.get('error', 'Unknown error')}")
        return ui_components
    
    # Komponen kritis yang harus ada - sesuai dengan DependencyInstallerInitializer._get_critical_components
    critical_components = [
        'ui', 
        'install_button', 
        'status', 
        'log_output', 
        'progress_container',
        'status_panel'
    ]
    
    # Validasi keberadaan komponen kritis
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    
    if missing_components:
        error_msg = f"Komponen UI tidak lengkap: {', '.join(missing_components)}"
        logger.error(error_msg)
        return {
            'error': 'Failed to validate UI components',
            'details': error_msg,
            'available_components': list(ui_components.keys())
        }
    
    # Validasi fungsi tombol install
    if not hasattr(ui_components['install_button'], 'on_click'):
        error_msg = "Install button tidak memiliki handler on_click"
        logger.error(error_msg)
        return {
            'error': 'Failed to validate UI components',
            'details': error_msg
        }
    
    # Validasi handler tombol install - skip jika tidak ada _click_handlers
    # Ini karena handler akan ditambahkan setelah validasi
    if hasattr(ui_components['install_button'], '_click_handlers') and \
       not ui_components['install_button']._click_handlers.callbacks:
        logger.warning("Install button tidak memiliki callback, akan ditambahkan nanti")
    
    # Semua validasi berhasil
    logger.info("UI components validation successful")
    
    # Pastikan module_name dan logger_namespace tersedia
    if 'module_name' not in ui_components:
        ui_components['module_name'] = MODULE_LOGGER_NAME
    if 'logger_namespace' not in ui_components:
        ui_components['logger_namespace'] = DEPENDENCY_INSTALLER_LOGGER_NAMESPACE
    
    return ui_components  # Kembalikan ui_components jika valid

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
