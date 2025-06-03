"""
File: smartcash/ui/setup/dependency_installer/utils/status_utils.py
Deskripsi: Utilitas untuk mengelola status panel dan notifikasi di dependency installer
"""

from typing import Dict, Any, Optional, Union
import re

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config

def highlight_numeric_params(text: str) -> str:
    """Highlight parameter numerik dalam teks dengan warna
    
    Args:
        text: Teks yang akan dihighlight
        
    Returns:
        Teks yang sudah dihighlight dengan parameter numerik berwarna hijau
    """
    # Pattern untuk mencocokkan parameter numerik
    pattern = r'(\d+(\.\d+)?%?)'
    
    # Warna untuk highlight dari konstanta terpusat
    colors = {
        'high': COLORS.get('success', '#28a745'),    # Hijau untuk nilai tinggi (>80%)
        'medium': COLORS.get('warning', '#ffc107'),  # Kuning untuk nilai sedang (40-80%)
        'low': COLORS.get('danger', '#dc3545')       # Merah untuk nilai rendah (<40%)
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

def update_status_panel(ui_components: Dict[str, Any], level: str = "info", message: str = "") -> None:
    """Update status panel dengan pesan dan level yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        level: Level status (info, success, warning, error, danger, debug, critical)
        message: Pesan yang akan ditampilkan
    """
    # Gunakan konfigurasi status dari constants.py
    config = get_status_config(level)
    
    # Pastikan pesan sudah memiliki emoji, jika belum tambahkan
    if not any(emoji in message for emoji in ['✅', '❌', '⚠️', 'ℹ️', '🔍', '📦']):
        message = f"{config['emoji']} {message}"
    
    # Update status panel jika tersedia
    if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
        ui_components['status_panel'].value = f'''
        <div style="padding:10px; background-color:{config['bg']}; 
                 color:{config['color']}; border-radius:5px; margin-bottom:10px;
                 border-left:4px solid {config['border']};">
            <p style="margin:0;">{message}</p>
        </div>
        '''
    
    # Update status container jika tersedia
    if 'status_container' in ui_components and hasattr(ui_components['status_container'], 'layout'):
        ui_components['status_container'].layout.visibility = 'visible'
        ui_components['status_container'].layout.border = f"1px solid {config['border']}"
    
    # Log pesan jika tersedia log_message
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"{config['emoji']} {message}", level)

def setup_status_utils(ui_components: Dict[str, Any]) -> None:
    """Setup fungsi-fungsi untuk status panel dan notifikasi
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Tambahkan fungsi update_status_panel ke ui_components dengan parameter yang benar
    ui_components['update_status_panel'] = lambda level="info", message="": update_status_panel(ui_components, level, message)
    
    # Tambahkan fungsi highlight_numeric_params ke ui_components
    ui_components['highlight_numeric_params'] = highlight_numeric_params
