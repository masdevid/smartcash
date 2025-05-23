"""
File: smartcash/ui/handlers/config_handlers.py
Deskripsi: Handler untuk konfigurasi umum
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML
import yaml
import os
from pathlib import Path
from smartcash.common.config import get_config_manager

def save_config(ui_components: Dict[str, Any], config: Dict[str, Any], 
               config_path: str, update_func: Callable = None, 
               component_name: str = "Konfigurasi") -> None:
    """
    Simpan konfigurasi.
    
    Args:
        ui_components: Komponen UI
        config: Konfigurasi yang akan disimpan
        config_path: Path atau nama module konfigurasi
        update_func: Fungsi untuk update konfigurasi dari UI
        component_name: Nama komponen untuk pesan status
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Update config dari UI jika ada fungsi update
        if update_func:
            config = update_func(config)
            
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Ekstrak nama modul jika config_path adalah path file
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            module_name = config_path.split('/')[-1].split('_')[0]
            if module_name.endswith('.yaml') or module_name.endswith('.yml'):
                module_name = module_name.rsplit('.', 1)[0]
        else:
            module_name = config_path
            
        # Simpan ke module config
        success = config_manager.save_module_config(module_name, config)
        
        # Tampilkan pesan sukses
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:green'>✅ {component_name} berhasil disimpan</p>"))
        
        if logger:
            logger.info(f"✅ {component_name} berhasil disimpan menggunakan SimpleConfigManager")
            
    except Exception as e:
        # Tampilkan pesan error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:red'>❌ Error saat menyimpan {component_name}: {str(e)}</p>"))
        
        if logger:
            logger.error(f"❌ Error saat menyimpan {component_name}: {str(e)}")

def reset_config(ui_components: Dict[str, Any], config: Dict[str, Any], 
                default_config: Dict[str, Any], update_ui_func: Callable = None, 
                component_name: str = "Konfigurasi") -> None:
    """
    Reset konfigurasi ke default.
    
    Args:
        ui_components: Komponen UI
        config: Konfigurasi yang akan direset
        default_config: Konfigurasi default
        update_ui_func: Fungsi untuk update UI dari konfigurasi
        component_name: Nama komponen untuk pesan status
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Reset config ke default
        for key, value in default_config.items():
            if key in config:
                if isinstance(value, dict) and isinstance(config[key], dict):
                    # Untuk nested dict, update value yang ada
                    config[key].update(value)
                else:
                    # Untuk value sederhana, langsung replace
                    config[key] = value
            else:
                # Tambahkan key baru jika belum ada
                config[key] = value
        
        # Update UI jika ada fungsi update
        if update_ui_func:
            update_ui_func()
        
        # Tampilkan pesan sukses
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:green'>✅ {component_name} berhasil direset ke default</p>"))
        
        if logger:
            logger.info(f"✅ {component_name} berhasil direset ke default")
            
    except Exception as e:
        # Tampilkan pesan error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:red'>❌ Error reset {component_name}: {str(e)}</p>"))
        
        if logger:
            logger.error(f"❌ Error reset {component_name}: {str(e)}")

def register_observers(ui_components: Dict[str, Any], components_list: list, 
                      handler_func: Callable) -> None:
    """
    Register observer untuk list komponen UI.
    
    Args:
        ui_components: Komponen UI
        components_list: List nama komponen yang akan diregister
        handler_func: Fungsi handler untuk observer
    """
    for comp_name in components_list:
        if comp_name in ui_components:
            ui_components[comp_name].observe(handler_func)

def unregister_observers(ui_components: Dict[str, Any], components_list: list, 
                        handler_func: Callable) -> None:
    """
    Unregister observer dari list komponen UI.
    
    Args:
        ui_components: Komponen UI
        components_list: List nama komponen yang akan diunregister
        handler_func: Fungsi handler untuk observer
    """
    for comp_name in components_list:
        if comp_name in ui_components:
            ui_components[comp_name].unobserve(handler_func)
