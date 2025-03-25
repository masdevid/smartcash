"""
File: smartcash/ui/info_boxes/environment_info.py
Deskripsi: Konten info box untuk konfigurasi environment
"""

import ipywidgets as widgets

def get_environment_info(open_by_default: bool = False) -> widgets.Accordion:
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS

    TITLE = "Tentang Konfigurasi Environment"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Environment setup dibutuhkan untuk:</p>
    <ul>
        <li><strong>Google Drive</strong>: Penyimpanan dataset, model, dan konfigurasi</li>
        <li><strong>Direktori Lokal</strong>: Struktur direktori yang konsisten untuk SmartCash</li>
    </ul>
    <p>Struktur direktori utama:</p>
    <ul>
        <li><code>data/</code>: Dataset dan hasil preprocessing</li>
        <li><code>configs/</code>: File konfigurasi</li>
        <li><code>runs/</code>: Hasil training dan evaluasi</li>
        <li><code>logs/</code>: File log</li>
        <li><code>checkpoints/</code>: Model checkpoint</li>
    </ul>
    
    <h4 style="color:inherit">{ICONS['folder']} Direktori Penting</h4>
    <p>Beberapa direktori yang perlu dibuat:</p>
    <ul>
        <li><code>data/train/images</code> dan <code>data/train/labels</code></li>
        <li><code>data/valid/images</code> dan <code>data/valid/labels</code></li>
        <li><code>data/test/images</code> dan <code>data/test/labels</code></li>
        <li><code>data/preprocessed</code></li>
    </ul>
    
    <h4 style="color:inherit">{ICONS['drive']} Google Drive Integration</h4>
    <p>Saat bekerja di Google Colab, SmartCash akan:</p>
    <ul>
        <li>Membuat symlinks antara direktori lokal dan Google Drive</li>
        <li>Sinkronisasi konfigurasi antara lokal dan Google Drive</li>
        <li>Menyimpan dataset dan model di Google Drive untuk persistensi</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", "⚙️", open_by_default)