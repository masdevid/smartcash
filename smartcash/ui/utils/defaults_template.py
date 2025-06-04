"""
File: smartcash/ui/utils/defaults_template.py
Deskripsi: Template untuk file defaults.py yang digunakan di modul UI
"""

from typing import Dict, Any

# Contoh konfigurasi default untuk modul UI
DEFAULT_CONFIG = {
    # Konfigurasi umum
    "general": {
        "module_name": "nama_modul",
        "description": "Deskripsi modul",
        "version": "1.0.0"
    },
    
    # Konfigurasi UI
    "ui": {
        "layout": "vertical",  # vertical, horizontal, tabs
        "show_status": True,
        "show_logger": True,
        "auto_update": False
    },
    
    # Konfigurasi parameter
    "parameters": {
        # Parameter spesifik modul
        "param1": 10,
        "param2": "nilai_default",
        "param3": True,
        
        # Nested parameters
        "nested": {
            "sub_param1": 5,
            "sub_param2": "nilai_nested"
        }
    }
}

# Fungsi untuk mendapatkan konfigurasi default
def get_default_config() -> Dict[str, Any]:
    """Mendapatkan konfigurasi default untuk modul"""
    return DEFAULT_CONFIG.copy()
