"""
File: smartcash/ui/setup/dependency_installer/utils/status_utils.py
Deskripsi: Utilitas untuk mengelola status panel dan notifikasi di dependency installer
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display
import re

def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """Update status panel dengan tipe dan pesan tertentu
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    # Mapping tipe status ke warna dan ikon
    status_map = {
        'info': {'color': '#17a2b8', 'icon': 'ℹ️'},
        'success': {'color': '#28a745', 'icon': '✅'},
        'warning': {'color': '#ffc107', 'icon': '⚠️'},
        'error': {'color': '#dc3545', 'icon': '❌'}
    }
    
    # Gunakan info sebagai fallback jika tipe tidak valid
    status_info = status_map.get(status_type, status_map['info'])
    
    # Update status widget
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
        ui_components['status_widget'].value = f"{status_info['icon']} {message}"
        if hasattr(ui_components['status_widget'], 'style'):
            ui_components['status_widget'].style.background_color = status_info['color']
            ui_components['status_widget'].style.color = 'white'
    
    # Update status container
    if 'status_container' in ui_components and hasattr(ui_components['status_container'], 'layout'):
        ui_components['status_container'].layout.visibility = 'visible'
        ui_components['status_container'].layout.border = f"1px solid {status_info['color']}"
    
    # Log pesan jika tersedia log_message
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"{status_info['icon']} {message}", status_type)

def highlight_numeric_params(text: str) -> str:
    """Highlight parameter numerik dalam text dengan warna
    
    Args:
        text: Text yang akan di-highlight
        
    Returns:
        Text dengan parameter numerik yang sudah di-highlight
    """
    # Pattern untuk mencocokkan parameter numerik
    pattern = r'(\d+(\.\d+)?%?)'
    
    # Warna untuk highlight
    colors = {
        'high': '#28a745',    # Hijau untuk nilai tinggi (>80%)
        'medium': '#ffc107',  # Kuning untuk nilai sedang (40-80%)
        'low': '#dc3545'      # Merah untuk nilai rendah (<40%)
    }
    
    def color_match(match):
        value = match.group(1)
        
        # Coba ekstrak nilai numerik
        try:
            if '%' in value:
                num_value = float(value.replace('%', ''))
                
                # Tentukan warna berdasarkan nilai persentase
                if num_value > 80:
                    color = colors['high']
                elif num_value > 40:
                    color = colors['medium']
                else:
                    color = colors['low']
            else:
                num_value = float(value)
                
                # Untuk nilai non-persentase, gunakan hijau untuk semua
                color = colors['high']
            
            # Return nilai dengan warna
            return f'<span style="color:{color};font-weight:bold">{value}</span>'
        except:
            # Jika gagal parsing, return nilai asli
            return value
    
    # Replace semua match dengan versi berwarna
    return re.sub(pattern, color_match, text)

def setup_status_utils(ui_components: Dict[str, Any]) -> None:
    """Setup fungsi-fungsi untuk status panel dan notifikasi
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Tambahkan fungsi update_status_panel ke ui_components
    ui_components['update_status_panel'] = lambda status_type, message: update_status_panel(ui_components, status_type, message)
    
    # Tambahkan fungsi highlight_numeric_params ke ui_components
    ui_components['highlight_numeric_params'] = highlight_numeric_params
