"""
File: smartcash/ui/info_boxes/dependencies_info.py
Deskripsi: Konten info box untuk informasi instalasi dependencies
"""

import ipywidgets as widgets

def get_dependencies_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Buat info box untuk penjelasan instalasi dependencies.
    
    Args:
        open_by_default: Apakah infobox terbuka secara default
        
    Returns:
        Widget Accordion berisi info box
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Tentang Instalasi Dependencies"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>SmartCash membutuhkan beberapa package Python untuk menjalankan deteksi mata uang. Package dibagi menjadi beberapa kategori:</p>
    
    <h4 style="color:inherit">{ICONS.get('package', '📦')} Core Packages</h4>
    <ul>
        <li><strong>YOLOv5 Requirements</strong>: Library dasar untuk model YOLOv5 (numpy, opencv, pyyaml, dll)</li>
        <li><strong>SmartCash Utils</strong>: Utility khusus untuk SmartCash (yaml, roboflow, dll)</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS.get('model', '🧠')} AI & ML Packages</h4>
    <ul>
        <li><strong>PyTorch</strong>: Deep learning framework untuk training dan inference</li>
        <li><strong>Albumentations</strong>: Library untuk augmentasi gambar saat training</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS.get('chart', '📊')} Visualization Packages</h4>
    <ul>
        <li><strong>Matplotlib & Pandas</strong>: Visualisasi hasil dan analisis data</li>
        <li><strong>Jupyter Tools</strong>: Widget dan tools untuk notebook environment</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS.get('info', 'ℹ️')} Alur Instalasi</h4>
    <p>Sistem melakukan instalasi dengan alur 3 tahap:</p>
    <ol>
        <li><strong>Deteksi</strong>: Memeriksa package yang sudah terinstall di sistem</li>
        <li><strong>Filter</strong>: Mengidentifikasi package yang perlu diinstall/update</li>
        <li><strong>Instalasi</strong>: Menginstall hanya package yang dibutuhkan</li>
    </ol>
    <p>Package yang sudah terinstall dengan benar akan ditandai ✅ dan tidak akan diinstall ulang kecuali perlu update versi.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", "📦", open_by_default)