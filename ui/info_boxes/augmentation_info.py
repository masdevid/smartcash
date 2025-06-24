"""
File: smartcash/ui/info_boxes/augmentation_info.py
Deskripsi: Info box untuk augmentasi dengan fokus pada data preprocessed
"""

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.components import create_info_accordion
from smartcash.ui.utils.constants import ICONS

def get_augmentation_info(open_by_default: bool = False) -> widgets.Accordion:
    TITLE = "Panduan Augmentasi Dataset"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
   
    <h4 style="color:inherit">Alur Kerja Augmentasi</h4>
    <ol>
        <li><b>Preprocessing:</b> Jalankan cell preprocessing terlebih dahulu untuk menyiapkan dataset dengan format <code>rp_kelas_id</code></li>
        <li><b>Balancing:</b> Augmentasi akan menyeimbangkan jumlah sampel per kelas secara otomatis</li>
        <li><b>Visualisasi:</b> Lihat hasil augmentasi dan bandingkan dengan gambar asli</li>
    </ol>

    <h4 style="color:inherit">Jenis Augmentasi</h4>
    <ul>
        <li><b>Combined:</b> Kombinasi posisi dan pencahayaan (direkomendasikan)</li>
        <li><b>Position:</b> Variasi posisi seperti rotasi, flipping, dan scaling</li>
        <li><b>Lighting:</b> Variasi pencahayaan seperti brightness, contrast dan HSV</li>
        <li><b>Extreme Rotation:</b> Rotasi dengan sudut yang ekstrim (>30°)</li>
    </ul>

    <h4 style="color:inherit">Data Preprocessing dan Balancing</h4>
    <p>Proses augmentasi akan menggunakan data yang telah di-preprocessing dengan prefix <code>rp_</code> sebagai sumber. 
    File preprocessed akan memiliki format <code>rp_kelas_uniqueid</code>, yang memudahkan augmentasi untuk melakukan balancing kelas.
    Opsi <i>Balance Classes</i> akan secara otomatis menyeimbangkan jumlah sampel untuk setiap kelas.</p>

    <p><b>Catatan:</b> Hasil augmentasi akan disimpan dengan format <code>aug_kelas_uniqueid</code> di direktori preprocessed yang sama.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", "🎨", open_by_default)