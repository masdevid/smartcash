"""
Footer components for Colab UI.

This module contains footer components and info boxes for the Colab setup interface.
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_module_info_box() -> widgets.HTML:
    """Create Colab-specific info box for footer.
    
    Returns:
        Widget containing module information
    """
    info_content = """
    <div style='font-size: 0.9em; line-height: 1.5;'>
        <h4 style='margin-top: 0;'>Tentang Setup Colab</h4>
        <p>Modul ini membantu Anda menyiapkan lingkungan Google Colab untuk SmartCash.</p>
        <p><strong>Versi:</strong> 1.0.0</p>
        <p><strong>Persyaratan:</strong></p>
        <ul style='margin: 5px 0; padding-left: 20px;'>
            <li>Runtime Google Colab</li>
            <li>Akses Google Drive</li>
            <li>Python 3.7+</li>
        </ul>
    </div>
    """
    return widgets.HTML(info_content)

def create_module_tips_box() -> widgets.HTML:
    """Create Colab-specific tips box for footer.
    
    Returns:
        Widget containing helpful tips
    """
    tips_content = """
    <div style='font-size: 0.9em; line-height: 1.5;'>
        <h4 style='margin-top: 0;'>Tips Cepat</h4>
        <ul style='margin: 5px 0; padding-left: 20px;'>
            <li>Gakan deteksi otomatis untuk mengonfigurasi lingkungan Anda</li>
            <li>Mount Google Drive untuk menyimpan pekerjaan Anda</li>
            <li>Periksa log untuk informasi detail</li>
            <li>Klik ikon bantuan untuk informasi lebih lanjut</li>
        </ul>
    </div>
    """
    return widgets.HTML(tips_content)

# For backward compatibility
_create_module_info_box = create_module_info_box
_create_module_tips_box = create_module_tips_box
