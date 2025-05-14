"""
File: smartcash/ui/training_config/config_handler.py
Deskripsi: Handler untuk konfigurasi training model
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from IPython.display import display, HTML
import yaml
import os
from pathlib import Path

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.config import get_config_manager

def save_config(ui_components: Dict[str, Any], config: Dict[str, Any], 
                config_path: str, update_func: Callable, title: str = "Konfigurasi") -> None:
    """
    Simpan konfigurasi ke file YAML dengan update dari UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi saat ini
        config_path: Path file konfigurasi
        update_func: Fungsi untuk mengupdate config dari UI
        title: Judul untuk pesan status
    """
    logger = ui_components.get('logger')
    if logger: logger.debug(f"{ICONS['info']} Menyimpan konfigurasi {title}...")
    
    try:
        # Update config dari UI
        # Pastikan config adalah dictionary dan update_func mengembalikan dictionary
        if not isinstance(config, dict):
            config = {}
            
        updated_config = update_func(config)
        
        # Pastikan hasil update adalah dictionary
        if isinstance(updated_config, dict):
            config = updated_config
        else:
            raise TypeError(f"Update function harus mengembalikan dictionary, bukan {type(updated_config).__name__}")
        
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Simpan ke file
        config_manager.save_config(config_path, config)
        
        # Simpan referensi konfigurasi di ui_components untuk memastikan persistensi
        ui_components['config'] = config
        if logger: logger.debug(f"{ICONS['success']} Konfigurasi {title} berhasil disimpan di ui_components")
        
        # Tampilkan pesan sukses
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(
                    f"<div style='color:{COLORS['success']}; padding:5px;'>"
                    f"{ICONS['success']} {title} berhasil disimpan ke {config_path}</div>"
                ))
        
        # Log jika tersedia
        if 'logger' in ui_components:
            ui_components['logger'].info(f"✅ {title} berhasil disimpan ke {config_path}")
            
    except Exception as e:
        # Tampilkan error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(
                    f"<div style='color:{COLORS['danger']}; padding:5px;'>"
                    f"{ICONS['error']} Error menyimpan {title.lower()}: {str(e)}</div>"
                ))
        
        # Log jika tersedia
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error menyimpan {title.lower()}: {str(e)}")

def reset_config(ui_components: Dict[str, Any], config: Dict[str, Any], 
                default_config: Dict[str, Any], update_ui_func: Callable, 
                title: str = "Konfigurasi") -> None:
    """
    Reset konfigurasi ke nilai default dan update UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi saat ini
        default_config: Konfigurasi default
        update_ui_func: Fungsi untuk mengupdate UI dari config
        title: Judul untuk pesan status
    """
    try:
        # Reset config ke default
        for key in default_config:
            if key in config:
                if isinstance(default_config[key], dict) and isinstance(config[key], dict):
                    # Deep update untuk nested dict
                    for subkey in default_config[key]:
                        config[key][subkey] = default_config[key][subkey]
                else:
                    # Langsung update untuk nilai non-dict
                    config[key] = default_config[key]
            else:
                # Tambahkan key baru jika belum ada
                config[key] = default_config[key]
        
        # Update UI dari config
        update_ui_func()
        
        # Tampilkan pesan sukses
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(
                    f"<div style='color:{COLORS['success']}; padding:5px;'>"
                    f"{ICONS['success']} {title} berhasil direset ke nilai default</div>"
                ))
        
        # Log jika tersedia
        if 'logger' in ui_components:
            ui_components['logger'].info(f"✅ {title} berhasil direset ke nilai default")
            
    except Exception as e:
        # Tampilkan error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(
                    f"<div style='color:{COLORS['danger']}; padding:5px;'>"
                    f"{ICONS['error']} Error mereset {title.lower()}: {str(e)}</div>"
                ))
        
        # Log jika tersedia
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error mereset {title.lower()}: {str(e)}")
