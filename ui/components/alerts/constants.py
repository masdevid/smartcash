"""
File: smartcash/ui/components/alerts/constants.py
Deskripsi: Konstanta untuk komponen alert
"""

from typing import Dict, Any

# Warna dasar untuk alert
COLORS = {
    'primary': '#3498db',
    'secondary': '#95a5a6',
    'success': '#2ecc71',
    'info': '#3498db',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'light': '#f8f9fa',
    'dark': '#343a40',
}

# Gaya untuk tiap tipe alert
ALERT_STYLES: Dict[str, Dict[str, Any]] = {
    'info': {
        'icon': 'ℹ️',
        'bg_color': '#e7f5fe',
        'border_color': COLORS['info'],
        'text_color': COLORS['dark'],
        'title': 'Info'
    },
    'success': {
        'icon': '✅',
        'bg_color': '#e8f8f0',
        'border_color': COLORS['success'],
        'text_color': '#155724',
        'title': 'Berhasil'
    },
    'warning': {
        'icon': '⚠️',
        'bg_color': '#fff8e6',
        'border_color': COLORS['warning'],
        'text_color': '#856404',
        'title': 'Peringatan'
    },
    'error': {
        'icon': '❌',
        'bg_color': '#fdeded',
        'border_color': COLORS['danger'],
        'text_color': '#721c24',
        'title': 'Error'
    }
}
