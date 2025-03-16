"""
File: smartcash/ui/handlers/shared/config_handler.py
Deskripsi: Handler untuk manajemen konfigurasi
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.components.shared.alerts import create_status_indicator

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
            config_manager = get_config_manager()
            ui_components['config_manager'] = config_manager
        except ImportError:
            pass
    
    return ui_components

def update_config(
    ui_components: Dict[str, Any],
    config_updates: Dict[str, Any],
    save_to_file: bool = True,
    config_path: Optional[str] = None
) -> bool:
    """
    Update konfigurasi dan simpan ke file.
    
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
        config_manager.update_config(config_updates)
        
        # Simpan ke file jika diperlukan
        if save_to_file and config_path:
            config_manager.save_config(config_path)
        
        # Update ui_components config
        ui_components['config'] = config_manager.get_config()
        return True
    else:
        # Fallback: update config di ui_components
        if 'config' in ui_components:
            current_config = ui_components['config']
            
            # Update config secara flat
            for key, value in config_updates.items():
                if isinstance(value, dict) and key in current_config and isinstance(current_config[key], dict):
                    current_config[key].update(value)
                else:
                    current_config[key] = value
            
            # Simpan ke file jika diperlukan
            if save_to_file and config_path:
                try:
                    with open(config_path, 'w') as f:
                        yaml.dump(current_config, f)
                except Exception:
                    return False
                    
            return True
    
    return False

def load_config(
    ui_components: Dict[str, Any],
    config_path: str,
    output_widget_key: str = 'status'
) -> Dict[str, Any]:
    """
    Load konfigurasi dari file dan tampilkan status.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config_path: Path file konfigurasi
        output_widget_key: Key untuk output widget
        
    Returns:
        Dictionary berisi konfigurasi
    """
    config = {}
    output_widget = ui_components.get(output_widget_key)
    
    # Dapatkan config manager jika ada
    config_manager = ui_components.get('config_manager')
    
    if config_manager:
        try:
            config = config_manager.load_config(config_path)
            ui_components['config'] = config
            
            if output_widget:
                with output_widget:
                    display(create_status_indicator("success", f"✅ Konfigurasi dimuat dari {config_path}"))
        except Exception as e:
            if output_widget:
                with output_widget:
                    display(create_status_indicator("error", f"❌ Gagal memuat konfigurasi: {str(e)}"))
    else:
        # Fallback implementation
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                ui_components['config'] = config
                
                if output_widget:
                    with output_widget:
                        display(create_status_indicator("success", f"✅ Konfigurasi dimuat dari {config_path}"))
            else:
                if output_widget:
                    with output_widget:
                        display(create_status_indicator("warning", f"⚠️ File konfigurasi tidak ditemukan: {config_path}"))
        except Exception as e:
            if output_widget:
                with output_widget:
                    display(create_status_indicator("error", f"❌ Gagal memuat konfigurasi: {str(e)}"))
    
    return config

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
    else:
        # Fallback: manual traversal
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
    save_to_file: bool = False,
    config_path: Optional[str] = None
) -> bool:
    """
    Set nilai dalam konfigurasi dengan dot notation.
    
    Args:
        ui_components: Dictionary berisi widget UI
        key_path: Path key dengan dot notation (e.g., 'training.batch_size')
        value: Nilai yang akan diset
        save_to_file: Simpan ke file setelah update
        config_path: Path file konfigurasi (opsional)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Check if config manager exists
    config_manager = ui_components.get('config_manager')
    
    if config_manager:
        config_manager.set(key_path, value)
        
        # Save to file if needed
        if save_to_file and config_path:
            config_manager.save_config(config_path)
            
        return True
    else:
        # Fallback: manual traversal and update
        config = ui_components.get('config', {})
        keys = key_path.split('.')
        
        # Create nested dictionaries if needed
        current = config
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
        
        # Save to file if needed
        if save_to_file and config_path:
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            except Exception:
                return False
                
        return True