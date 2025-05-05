"""
File: smartcash/ui/info_boxes/backbones_info.py
Deskripsi: Konten info box untuk konfigurasi backbones
"""

import ipywidgets as widgets

def get_backbones_info(open_by_default: bool = False) -> widgets.Accordion:
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Tentang Konfigurasi Backbones"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Backbones adalah model yang digunakan sebagai basis untuk pelatihan model:</p>
    <ul>
        <li><strong>Backbone</strong>: Model yang digunakan sebagai basis untuk pelatihan model</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", "⚙️", open_by_default)
