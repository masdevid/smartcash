"""File: smartcash/ui/setup/dependency_installer/utils/constants.py
Deskripsi: Konstanta terpusat untuk dependency installer UI
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS

# Status mapping yang konsisten untuk seluruh dependency installer
STATUS_CONFIG = {
    "info": {
        "bg": "#d1ecf1", 
        "color": "#0c5460", 
        "border": COLORS.get('info', "#17a2b8"), 
        "emoji": ICONS.get('info', "ℹ️"),
        "text": "Informasi"
    },
    "success": {
        "bg": "#d4edda", 
        "color": "#155724", 
        "border": COLORS.get('success', "#28a745"), 
        "emoji": ICONS.get('success', "✅"),
        "text": "Terinstall"
    },
    "warning": {
        "bg": "#fff3cd", 
        "color": "#856404", 
        "border": COLORS.get('warning', "#ffc107"), 
        "emoji": ICONS.get('warning', "⚠️"),
        "text": "Perlu update"
    },
    "error": {
        "bg": "#f8d7da", 
        "color": "#721c24", 
        "border": COLORS.get('danger', "#dc3545"), 
        "emoji": ICONS.get('error', "❌"),
        "text": "Tidak terinstall"
    },
    "danger": {
        "bg": "#f8d7da", 
        "color": "#721c24", 
        "border": COLORS.get('danger', "#dc3545"), 
        "emoji": ICONS.get('error', "❌"),
        "text": "Error"
    },
    "debug": {
        "bg": "#e2e3e5", 
        "color": "#383d41", 
        "border": COLORS.get('secondary', "#6c757d"), 
        "emoji": ICONS.get('debug', "🔍"),
        "text": "Debug"
    },
    "critical": {
        "bg": "#f8d7da", 
        "color": "#721c24", 
        "border": COLORS.get('danger', "#dc3545"), 
        "emoji": "🚨",
        "text": "Kritis"
    }
}

# Mapping untuk status package
PACKAGE_STATUS = {
    "success": {"icon": ICONS.get('success', "✅"), "color": COLORS.get('success', "#28a745"), "text": "Terinstall"},
    "warning": {"icon": ICONS.get('warning', "⚠️"), "color": COLORS.get('warning', "#ffc107"), "text": "Perlu update"},
    "error": {"icon": ICONS.get('error', "❌"), "color": COLORS.get('danger', "#dc3545"), "text": "Tidak terinstall"},
    "info": {"icon": ICONS.get('debug', "🔍"), "color": COLORS.get('info', "#17a2b8"), "text": "Checking..."}
}

# Default ke info jika level tidak valid
def get_status_config(level: str) -> Dict[str, Any]:
    """Mendapatkan konfigurasi status berdasarkan level
    
    Args:
        level: Level status (info, success, warning, error, danger, debug, critical)
        
    Returns:
        Dictionary konfigurasi status
    """
    return STATUS_CONFIG.get(level, STATUS_CONFIG["info"])

# Default ke info jika status tidak valid
def get_package_status(status: str) -> Dict[str, Any]:
    """Mendapatkan konfigurasi status package berdasarkan status
    
    Args:
        status: Status package (success, warning, error, info)
        
    Returns:
        Dictionary konfigurasi status package
    """
    return PACKAGE_STATUS.get(status, PACKAGE_STATUS["info"])
