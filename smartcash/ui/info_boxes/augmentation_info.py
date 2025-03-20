"""
File: smartcash/ui/info_boxes/augmentation_info.py
Deskripsi: Konten info box untuk augmentasi dataset
"""

import ipywidgets as widgets
from smartcash.ui.utils.info_utils import create_info_accordion

TITLE = "Tentang Augmentasi Dataset"

def get_augmentation_info(open_by_default: bool = False) -> widgets.Accordion:
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
   
    <p>Augmentasi data adalah teknik untuk meningkatkan jumlah dan variasi data dengan menerapkan transformasi yang mempertahankan informasi label.</p>
    
    <h4 style="color:inherit">Jenis Augmentasi</h4>
    <ul>
        <li><strong>Combined (Recommended)</strong>: Kombinasi beberapa transformasi sekaligus</li>
        <li><strong>Position Variations</strong>: Rotasi, flip, translasi, dan scaling</li>
        <li><strong>Lighting Variations</strong>: Perubahan brightness, contrast, dan saturation</li>
        <li><strong>Extreme Rotation</strong>: Rotasi dengan sudut besar (30-90°)</li>
    </ul>
    
    <h4 style="color:inherit">Parameter Position</h4>
    <ul>
        <li><strong>Flip prob</strong>: Probabilitas flip horizontal (0-1)</li>
        <li><strong>Flipud prob</strong>: Probabilitas flip vertical (0-1)</li>
        <li><strong>Degrees</strong>: Derajat maksimum rotasi (0-90)</li>
        <li><strong>Translate</strong>: Nilai maksimum translasi sebagai fraksi dari ukuran (0-0.5)</li>
        <li><strong>Scale</strong>: Nilai maksimum scaling sebagai fraksi dari ukuran (0-0.5)</li>
        <li><strong>Shear</strong>: Nilai maksimum shearing dalam derajat (0-20)</li>
    </ul>
    
    <h4 style="color:inherit">Parameter Lighting</h4>
    <ul>
        <li><strong>HSV Hue</strong>: Pergeseran hue dalam format HSV (0-0.1)</li>
        <li><strong>HSV Sat</strong>: Perubahan saturation dalam format HSV (0-1)</li>
        <li><strong>HSV Value</strong>: Perubahan value/brightness dalam format HSV (0-1)</li>
        <li><strong>Contrast</strong>: Perubahan contrast gambar (0-1)</li>
        <li><strong>Brightness</strong>: Perubahan brightness gambar (0-1)</li>
        <li><strong>Compress</strong>: Efek kompresi/blur gambar (0-1)</li>
    </ul>
    
    <h4 style="color:inherit">Tips Augmentasi</h4>
    <ul>
        <li>Gunakan "Combined" untuk keseimbangan terbaik antara variasi dan kecepatan</li>
        <li>Tentukan jumlah variasi (2-3) yang sesuai dengan ukuran dataset</li>
        <li>Augmentasi terbaik diterapkan pada split train saja (bukan validation/test)</li>
        <li>Opsi "Process bboxes" memastikan koordinat bounding box ikut diubah sesuai augmentasi</li>
        <li>Aktifkan "Validate results" untuk memeriksa integritas file hasil augmentasi</li>
    </ul>
    
    <p><strong>📝 Konfigurasi</strong> akan otomatis disimpan ke <code>configs/augmentation_config.yaml</code></p>
    """
    
    return create_info_accordion(TITLE, content, "info", "🎨", open_by_default)