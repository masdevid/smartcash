"""
File: smartcash/ui/info_boxes/augmentation_info.py
Deskripsi: Konten info box untuk augmentasi dataset dengan informasi yang lebih sederhana
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion

TITLE = "Tentang Augmentasi Dataset"

def get_augmentation_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
   
    <p>Augmentasi dataset adalah teknik untuk meningkatkan variasi dan ukuran dataset dengan menambahkan data baru yang dimodifikasi. Tersedia beberapa jenis augmentasi:</p>
    
    <ul>
        <li><strong>Combined</strong>: Gabungan dari augmentasi position dan lighting untuk variasi yang beragam</li>
        <li><strong>Position Variations</strong>: Variasi posisi dengan flip, rotasi, dan pergeseran geometri</li>
        <li><strong>Lighting Variations</strong>: Variasi pencahayaan dengan perubahan brightness, contrast, dan saturasi</li>
        <li><strong>Extreme Rotation</strong>: Rotasi sudut ekstrem (30-90°) untuk latihan yang lebih robust</li>
    </ul>
    
    <p><strong>Variations</strong>: Jumlah variasi gambar yang akan dibuat untuk setiap gambar asli</p>
    <p><strong>Prefix</strong>: Nama awalan untuk file hasil augmentasi (default: "aug")</p>
    <p><strong>Process bboxes</strong>: Proses juga bounding box bersamaan dengan gambar</p>
    <p><strong>Validate results</strong>: Validasi hasil augmentasi untuk memastikan integritas</p>
    
    <p><strong>📝 Catatan</strong>: Hasil augmentasi akan disimpan ke direktori output (default: data/augmented)</p>
    """
    
    return create_info_accordion(TITLE, content, "info", "🎨", open_by_default)