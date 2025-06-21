"""
File: smartcash/common/io/serialization.py
Deskripsi: Utilitas untuk serialisasi dan deserialisasi data (JSON, YAML, dll)
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

from smartcash.common.io.path_utils import ensure_dir

def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load data dari file JSON.
    
    Args:
        path: Path file JSON
        default: Nilai default jika file tidak ada atau invalid
        
    Returns:
        Data dari file JSON atau default
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(
    data: Any, 
    path: Union[str, Path], 
    pretty: bool = True, 
    create_dirs: bool = True
) -> bool:
    """
    Simpan data ke file JSON.
    
    Args:
        data: Data yang akan disimpan
        path: Path file JSON
        pretty: Format dengan indentasi
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        path = Path(path)
        
        if create_dirs:
            ensure_dir(path.parent)
        
        with open(path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        return True
    except Exception:
        return False

def load_yaml(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load data dari file YAML.
    
    Args:
        path: Path file YAML
        default: Nilai default jika file tidak ada atau invalid
        
    Returns:
        Data dari file YAML atau default
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or default
    except (FileNotFoundError, yaml.YAMLError):
        return default

def save_yaml(
    data: Any, 
    path: Union[str, Path], 
    default_flow_style: bool = False,
    create_dirs: bool = True
) -> bool:
    """
    Simpan data ke file YAML.
    
    Args:
        data: Data yang akan disimpan
        path: Path file YAML
        default_flow_style: Format YAML flow style
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        path = Path(path)
        
        if create_dirs:
            ensure_dir(path.parent)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, allow_unicode=True)
        return True
    except Exception:
        return False

def load_config(
    path: Union[str, Path], 
    default_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Load konfigurasi dari file (YAML atau JSON).
    
    Args:
        path: Path file konfigurasi
        default_config: Konfigurasi default jika file tidak ada
        
    Returns:
        Dictionary konfigurasi
    """
    path = Path(path)
    
    if not path.exists():
        return default_config or {}
    
    if path.suffix.lower() in ['.yml', '.yaml']:
        return load_yaml(path, default_config)
    elif path.suffix.lower() == '.json':
        return load_json(path, default_config)
    else:
        # Coba sebagai YAML default
        result = load_yaml(path, None)
        if result is not None:
            return result
        
        # Coba sebagai JSON default
        result = load_json(path, None)
        if result is not None:
            return result
            
        # Jika kedua-duanya gagal
        return default_config or {}

def save_config(
    config: Dict[str, Any], 
    path: Union[str, Path], 
    format: str = None,
    create_dirs: bool = True
) -> bool:
    """
    Simpan konfigurasi ke file (YAML atau JSON).
    
    Args:
        config: Dictionary konfigurasi
        path: Path file konfigurasi
        format: Format file ('yaml', 'json') atau auto-detect dari ekstensi
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    path = Path(path)
    
    # Auto-detect format
    if format is None:
        ext = path.suffix.lower()
        if ext in ['.yml', '.yaml']:
            format = 'yaml'
        elif ext == '.json':
            format = 'json'
        else:
            # Default to YAML
            format = 'yaml'
    
    # Simpan sesuai format
    if format.lower() == 'yaml':
        return save_yaml(config, path, create_dirs=create_dirs)
    elif format.lower() == 'json':
        return save_json(config, path, pretty=True, create_dirs=create_dirs)
    else:
        raise ValueError(f"Format tidak didukung: {format}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gabungkan dua konfigurasi, dengan override sebagai prioritas tinggi.
    
    Args:
        base_config: Konfigurasi dasar
        override_config: Konfigurasi yang akan menimpa base_config
        
    Returns:
        Dictionary konfigurasi gabungan
    """
    import copy
    result = copy.deepcopy(base_config)
    
    # Recursive merge for dictionaries
    def _merge_recursive(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = _merge_recursive(base[key], value)
            else:
                base[key] = value
        return base
    
    return _merge_recursive(result, override_config)

def load_configs_from_directory(
    directory: Union[str, Path], 
    pattern: str = '*.yaml', 
    base_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Load dan merge semua konfigurasi dari direktori.
    
    Args:
        directory: Direktori yang berisi file konfigurasi
        pattern: Pattern untuk filter file
        base_config: Konfigurasi dasar awal
        
    Returns:
        Dictionary konfigurasi gabungan
    """
    from smartcash.common.io.path_utils import find_files
    
    directory = Path(directory)
    config_files = find_files(directory, [pattern])
    
    result = base_config or {}
    
    for config_file in config_files:
        config = load_config(config_file)
        result = merge_configs(result, config)
    
    return result