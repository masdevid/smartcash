"""
File: smartcash/ui/info_boxes/preprocessing_info.py
Deskripsi: Konten info box untuk preprocessing
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion

TITLE = "Tentang Preprocessing"

def get_preprocessing_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
   
    <p>Preprocessing meliputi beberapa langkah penting:</p>
    <ul>
        <li><strong>Resize</strong>: Ubah ukuran gambar menjadi ukuran yang seragam</li>
        <li><strong>Normalization</strong>: Normalisasi pixel values untuk training yang lebih stabil</li>
        <li><strong>Caching</strong>: Simpan gambar yang sudah diproses untuk mempercepat loading</li>
        <li><strong>Validation</strong>: Validasi integritas dataset, cek label dan gambar rusak</li>
    </ul>
    <p><strong>📝 Konfigurasi</strong> akan otomatis disimpan ke <code>configs/preprocessing_config.yaml</code></p>
    
    <p><strong>Catatan:</strong> Gambar yang telah dipreprocessing akan disimpan di direktori 
    <code>data/preprocessed/[split]</code> untuk penggunaan berikutnya.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", "📄", open_by_default)
