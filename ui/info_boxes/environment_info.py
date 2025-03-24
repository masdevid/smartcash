"""
File: smartcash/ui/info_boxes/environment_info.py
Deskripsi: Konten info box untuk konfigurasi environment SmartCash
"""

import ipywidgets as widgets

def get_environment_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Buat info box berisi panduan environment.
    
    Args:
        open_by_default: Flag untuk membuka accordion secara default
        
    Returns:
        Widget accordion berisi info environment
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Tentang Environment SmartCash"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>SmartCash dapat berjalan di dua environment:</p>
    <ul>
        <li><strong>Google Colab</strong>: Environment cloud dengan dukungan GPU dan integrasi Google Drive.</li>
        <li><strong>Local</strong>: Environment lokal untuk development dan testing.</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS.get('folder', '📁')} Struktur Direktori</h4>
    <p>SmartCash membutuhkan struktur direktori berikut:</p>
    <pre style="background-color:#f8f9fa; padding:10px; border-radius:4px;">
SmartCash/
├── configs/          - File konfigurasi YAML 
├── data/             - Dataset
│   ├── train/        - Data training
│   ├── valid/        - Data validasi
│   ├── test/         - Data testing
│   └── preprocessed/ - Data hasil preprocessing
├── runs/             - Output training
├── logs/             - Log aplikasi
└── checkpoints/      - Model checkpoint
    </pre>
    
    <h4 style="color:inherit">{ICONS.get('drive', '🔄')} Integrasi Google Drive</h4>
    <p>Saat berjalan di Google Colab, SmartCash menggunakan Google Drive untuk:</p>
    <ul>
        <li>Penyimpanan dataset yang persisten</li>
        <li>Backup konfigurasi</li>
        <li>Penyimpanan model terlatih</li>
        <li>Symlinks otomatis ke direktori lokal</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS.get('config', '⚙️')} Konfigurasi Sinkronisasi</h4>
    <p>Konfigurasi disinkronkan otomatis dengan strategi:</p>
    <ul>
        <li><strong>Drive sebagai sumber kebenaran</strong>: Saat aplikasi pertama kali dijalankan</li>
        <li><strong>Merge dua arah</strong>: Saat ada perubahan konfigurasi</li>
        <li><strong>Backup lokal</strong>: Sebelum sinkronisasi untuk menjaga keamanan data</li>
    </ul>
    
    <div style="margin-top:10px; padding:10px; background-color:#f8f9fa; border-left:4px solid #17a2b8; border-radius:4px;">
        <p style="margin:0"><strong>Tip:</strong> Di Google Colab, tombol "Hubungkan Google Drive" akan otomatis mount Drive dan melakukan sinkronisasi konfigurasi.</p>
    </div>
    """
    
    return create_info_accordion(TITLE, content, "info", "⚙️", open_by_default)