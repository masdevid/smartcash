"""
File: smartcash/ui/handlers/config_handler.py
Deskripsi: Handler untuk manajemen konfigurasi dengan metode yang lebih efisien
"""

import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional, Union

def setup_config_handlers(
    ui_components: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi di UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi opsional
        
    Returns:
        Dictionary UI components yang telah ditambahkan config handler
    """
    # Attach config ke ui_components jika belum ada
    if config is not None and 'config' not in ui_components:
        ui_components['config'] = config
        
    # Coba tambahkan config manager jika belum ada
    if 'config_manager' not in ui_components:
        try:
            from smartcash.common.config import get_config_manager
            ui_components['config_manager'] = get_config_manager()
        except ImportError:
            pass
    
    return ui_components

def handle_config_save(ui_components: Dict[str, Any], config_path: str) -> bool:
    """
    Handler untuk menyimpan konfigurasi ke file.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    config_manager = ui_components.get('config_manager')
    config = ui_components.get('config', {})
    
    try:
        if config_manager:
            config_manager.save_config(config_path)
        else:
            # Buat direktori jika belum ada
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Simpan ke file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False

def handle_config_load(ui_components: Dict[str, Any], config_path: str, merge: bool = True) -> Dict[str, Any]:
    """
    Handler untuk memuat konfigurasi dari file.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config_path: Path file konfigurasi
        merge: Gabungkan dengan konfigurasi yang sudah ada
        
    Returns:
        Dictionary konfigurasi yang dimuat
    """
    config_manager = ui_components.get('config_manager')
    
    try:
        if config_manager:
            if merge:
                config = config_manager.merge_config(config_path)
            else:
                config = config_manager.load_config(config_path)
            ui_components['config'] = config
        else:
            # Load file secara manual
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                
                if merge and 'config' in ui_components:
                    # Merge konfigurasi
                    config = ui_components['config']
                    for k, v in loaded_config.items():
                        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                            config[k].update(v)
                        else:
                            config[k] = v
                else:
                    ui_components['config'] = loaded_config
                    config = loaded_config
            else:
                config = {}
        return config
    except Exception:
        if 'config' in ui_components:
            return ui_components['config']
        return {}

def update_config(
    ui_components: Dict[str, Any],
    config_updates: Dict[str, Any],
    save_to_file: bool = False,
    config_path: Optional[str] = None
) -> bool:
    """
    Update konfigurasi dan simpan ke file jika diperlukan.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config_updates: Dictionary berisi update konfigurasi
        save_to_file: Simpan ke file setelah update
        config_path: Path file konfigurasi (opsional)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Dapatkan config manager jika ada
    config_manager = ui_components.get('config_manager')
    
    if config_manager:
        # Update konfigurasi via config manager
        for key, value in config_updates.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    config_manager.set(f"{key}.{subkey}", subvalue)
            else:
                config_manager.set(key, value)
        
        # Simpan ke file jika diperlukan
        if save_to_file and config_path:
            config_manager.save_config(config_path)
        
        # Update ui_components config
        ui_components['config'] = config_manager.config
        return True
    else:
        # Fallback: update config di ui_components
        if 'config' in ui_components:
            config = ui_components['config']
            
            # Update config secara flat
            for key, value in config_updates.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
            
            # Simpan ke file jika diperlukan
            if save_to_file and config_path:
                try:
                    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                except Exception:
                    return False
                    
            return True
    
    return False

def get_config_value(
    ui_components: Dict[str, Any],
    key_path: str,
    default_value: Any = None
) -> Any:
    """
    Dapatkan nilai dari konfigurasi dengan dot notation.
    
    Args:
        ui_components: Dictionary berisi widget UI
        key_path: Path key dengan dot notation (e.g., 'training.batch_size')
        default_value: Nilai default jika key tidak ditemukan
        
    Returns:
        Nilai konfigurasi
    """
    # Check if config manager exists
    config_manager = ui_components.get('config_manager')
    
    if config_manager:
        return config_manager.get(key_path, default_value)
    
    # Manual traversal
    config = ui_components.get('config', {})
    keys = key_path.split('.')
    
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default_value
            
    return config

def set_config_value(
    ui_components: Dict[str, Any],
    key_path: str,
    value: Any,
    save: bool = False,
    config_path: Optional[str] = None
) -> bool:
    """
    Set nilai dalam konfigurasi dengan dot notation.
    
    Args:
        ui_components: Dictionary berisi widget UI
        key_path: Path key dengan dot notation (e.g., 'training.batch_size')
        value: Nilai yang akan diset
        save: Simpan ke file setelah update
        config_path: Path file konfigurasi (opsional)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    config_manager = ui_components.get('config_manager')
    
    if config_manager:
        config_manager.set(key_path, value)
        if save and config_path:
            config_manager.save_config(config_path)
        return True
    
    # Manual update
    config = ui_components.get('config', {})
    keys = key_path.split('.')
    
    # Create nested dictionaries
    current = config
    for i, key in enumerate(keys[:-1]):
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set value
    current[keys[-1]] = value
    
    # Save if needed
    if save and config_path:
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception:
            return False
    
    return True