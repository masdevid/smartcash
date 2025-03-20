"""
File: smartcash/ui/info_boxes/dependencies_info.py
Deskripsi: Konten info box untuk instalasi dependencies
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion

TITLE = "Tentang Package Installation"

def get_dependencies_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <p>SmartCash memerlukan beberapa package untuk berjalan dengan optimal. Package diurutkan instalasi dari kecil ke besar:</p>
    
    <ol>
        <li>Notebook tools (ipywidgets, tqdm)</li>
        <li>Utility packages (pyyaml, termcolor)</li>
        <li>Data processing (matplotlib, pandas)</li>
        <li>Computer vision (OpenCV, Albumentations)</li>
        <li>Machine learning (PyTorch)</li>
    </ol>
    
    <h4 style="color:inherit">Kategori Package</h4>
    <ul>
        <li><strong>Core Packages</strong>: YOLOv5 requirements, SmartCash utils, Notebook tools</li>
        <li><strong>ML Packages</strong>: PyTorch, OpenCV, Albumentations</li>
        <li><strong>Viz Packages</strong>: Matplotlib, Pandas, Seaborn</li>
    </ul>
    
    <p><strong>⚠️ Catatan:</strong> Instalasi PyTorch mungkin memerlukan waktu lebih lama, terutama di lingkungan dengan koneksi lambat.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", "📦", open_by_default)
