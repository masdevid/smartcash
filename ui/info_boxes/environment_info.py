"""
File: smartcash/ui/info_boxes/environment.py
Deskripsi: Konten info box untuk konfigurasi environment
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion

TITLE = "Tentang Konfigurasi Environment"
def get_environment_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <p>Konfigurasi environment akan memastikan project SmartCash berjalan dengan baik di lingkungan saat ini.</p>
    
    <h4 style="color:inherit">Google Colab</h4>
    <ul>
        <li>Hubungkan ke Google Drive untuk menyimpan dataset, model, dan hasil</li>
        <li>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></li>
        <li>Symlink akan dibuat untuk <code>data</code>, <code>configs</code>, <code>runs</code>, dll</li>
    </ul>
    
    <h4 style="color:inherit">Lingkungan Lokal</h4>
    <ul>
        <li>Pastikan struktur direktori telah dibuat</li>
        <li>Direktori utama: <code>data</code>, <code>configs</code>, <code>runs</code>, <code>logs</code></li>
        <li>Subdirektori data: <code>train</code>, <code>valid</code>, <code>test</code></li>
    </ul>
    
    <h4 style="color:inherit">Tips Konfigurasi</h4>
    <ul>
        <li>Simpan konfigurasi dalam file YAML di direktori <code>configs</code></li>
        <li>Sesuaikan <code>configs/colab_config.yaml</code> untuk Google Colab</li>
        <li>Gunakan <code>configs/base_config.yaml</code> untuk pengaturan umum</li>
        <li>Cache preprocessing dapat mempercepat loading dataset</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", "⚙️", open_by_default)
