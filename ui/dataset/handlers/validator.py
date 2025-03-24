"""
File: smartcash/ui/dataset/handlers/validator.py
Deskripsi: Utilitas untuk validasi konfigurasi download dataset
"""

from typing import Dict, Any, Tuple
from pathlib import Path

def validate_download_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi download dataset.
    
    Args:
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Cek tipe endpoint
    endpoint_type = config.get('type')
    if not endpoint_type:
        return False, "Endpoint tidak dipilih"
    
    # Validasi berdasarkan tipe endpoint
    if endpoint_type == 'roboflow':
        return _validate_roboflow_config(config)
    elif endpoint_type == 'drive':
        return _validate_drive_config(config)
    elif endpoint_type == 'url':
        return _validate_url_config(config)
    else:
        return False, f"Tipe endpoint tidak valid: {endpoint_type}"

def _validate_roboflow_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi Roboflow.
    
    Args:
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Cek field wajib
    required_fields = [
        ('workspace', 'Workspace ID'),
        ('project', 'Project ID'),
        ('version', 'Version'),
        ('api_key', 'API Key')
    ]
    
    for field, label in required_fields:
        if not config.get(field):
            return False, f"{label} wajib diisi"
    
    # Cek format version (harus angka atau angka dengan awalan 'v')
    version = config.get('version', '')
    if version:
        # Izinkan format 'v1', 'v1.0', '1', atau '1.0'
        if not (version.isdigit() or (version.startswith('v') and version[1:].replace('.', '').isdigit())):
            return False, f"Format version tidak valid: {version}"
    
    # Validasi output dir
    return _validate_output_config(config)

def _validate_drive_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi Google Drive.
    
    Args:
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Cek ketersediaan Google Drive
    try:
        from smartcash.common.environment import get_environment_manager
        env = get_environment_manager()
        if not env.is_drive_mounted:
            return False, "Google Drive tidak terpasang"
    except ImportError:
        # Fallback check
        import os
        if not os.path.exists('/content/drive/MyDrive'):
            return False, "Google Drive tidak terpasang"
    
    # Cek field wajib
    if not config.get('folder'):
        return False, "Folder Google Drive wajib diisi"
    
    # Validasi output dir
    return _validate_output_config(config)

def _validate_url_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi URL.
    
    Args:
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Cek URL
    url = config.get('url', '')
    if not url:
        return False, "URL wajib diisi"
    
    # Simple URL validation - cek protokol dan domain
    if not (url.startswith('http://') or url.startswith('https://')):
        return False, "URL harus dimulai dengan http:// atau https://"
    
    # Cek ekstensi file
    supported_extensions = ['.zip', '.tar', '.gz', '.tar.gz']
    if not any(url.lower().endswith(ext) for ext in supported_extensions):
        return False, "URL harus mengarah ke file arsip (.zip, .tar, .gz, or .tar.gz)"
    
    # Validasi output dir
    return _validate_output_config(config)

def _validate_output_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi output.
    
    Args:
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Cek output directory
    output_dir = config.get('output_dir')
    if not output_dir:
        return False, "Output directory wajib diisi"
    
    # Cek format output
    output_format = config.get('output_format')
    if not output_format:
        return False, "Format output wajib diisi"
    
    # Semua validasi berhasil
    return True, ""