"""
File: smartcash/ui/info_boxes/download_info.py
Deskripsi: Konten info box untuk download
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion
from smartcash.ui.utils.constants import ICONS

TITLE = "Panduan Download Dataset"
def get_download_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
   
    <h4 style="color:inherit">Cara Download Dataset</h4>
    <ol>
        <li><b>Roboflow:</b> Masukkan API key Roboflow dan konfigurasi workspace/project</li>
        <li><b>Local:</b> Upload file ZIP yang berisi dataset dalam format YOLOv5</li>
    </ol>
    <h4 style="color:inherit">Setup API Key Roboflow</h4>
    <p>Anda memiliki beberapa cara untuk menyediakan API key Roboflow:</p>
    <ol>
        <li><b>Google Secret (untuk Colab):</b>
            <ul>
                <li>Klik ikon kunci {ICONS['settings']} di sidebar kiri</li>
                <li>Tambahkan secret baru dengan nama <code>ROBOFLOW_API_KEY</code></li>
                <li>Masukkan API key Roboflow Anda sebagai nilai</li>
                <li>Klik "Save". API key akan diambil otomatis</li>
            </ul>
        </li>
        <li><b>Input Manual:</b> Masukkan API key secara langsung pada field yang tersedia</li>
        <li><b>Config File:</b> Tambahkan ke <code>configs/base_config.yaml</code></li>
    </ol>
    <h4 style="color:inherit">Backup Dataset</h4>
    <p>Jika opsi backup diaktifkan, dataset yang ada akan dibackup ke file ZIP sebelum diubah. File ZIP akan disimpan di direktori <code>data/backups</code>.</p>
    <h4 style="color:inherit">Struktur Dataset YOLOv5</h4>
    <pre>
data/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── test/
      ├── images/
      └── labels/
        </pre>
    """
    
    return create_info_accordion(TITLE, content, "info", "📄", open_by_default)
