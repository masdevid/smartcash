"""
File: smartcash/ui/utils/ui_logger_namespace.py
Deskripsi: Utilitas untuk manajemen namespace di UI logger untuk mencegah pencampuran log
"""

from typing import Dict, Any, Optional

# Daftar namespace yang diketahui dan ID unik mereka
KNOWN_NAMESPACES = {
    "smartcash.setup.dependency_installer": "DEP-INSTALLER",
    "smartcash.dataset.download": "DATASET-DOWNLOAD",
    "smartcash.ui.env_config": "ENV-CONFIG",
    "smartcash.dataset.preprocessing": "PREPROCESSING",
    "smartcash.dataset.augmentation": "AUGMENTATION"
    # Tambahkan namespace-namespace lain di sini jika diperlukan
}

# Export konstanta
DEPENDENCY_INSTALLER_LOGGER_NAMESPACE = "smartcash.setup.dependency_installer" 
DOWNLOAD_LOGGER_NAMESPACE = "smartcash.dataset.download"
ENV_CONFIG_LOGGER_NAMESPACE = "smartcash.ui.env_config"
PREPROCESSING_LOGGER_NAMESPACE = "smartcash.dataset.preprocessing"
AUGMENTATION_LOGGER_NAMESPACE = "smartcash.dataset.augmentation"

def get_namespace_id(ui_components: Dict[str, Any]) -> Optional[str]:
    """
    Dapatkan ID namespace dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ID namespace atau None jika tidak ditemukan
    """
    # Cek apakah namespace tersedia di ui_components
    namespace = ui_components.get('logger_namespace')
    
    # Jika namespace tersedia, dapatkan ID-nya
    if namespace:
        return KNOWN_NAMESPACES.get(namespace, namespace)
    
    # Cek flag spesifik yang disetel oleh inisializer
    if ui_components.get('dependency_installer_initialized', False):
        return KNOWN_NAMESPACES.get("smartcash.setup.dependency_installer")
    
    if ui_components.get('download_initialized', False):
        return KNOWN_NAMESPACES.get("smartcash.dataset.download")
    
    if ui_components.get('env_config_initialized', False):
        return KNOWN_NAMESPACES.get("smartcash.ui.env_config")
    
    # Default: tidak ada namespace yang dikenali
    return None

def format_log_message(ui_components: Dict[str, Any], message: str) -> str:
    """
    Format pesan log dengan ID namespace jika tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan diformat
        
    Returns:
        Pesan yang diformat dengan namespace
    """
    namespace_id = get_namespace_id(ui_components)
    
    if namespace_id:
        return f"[{namespace_id}] {message}"
    
    return message 