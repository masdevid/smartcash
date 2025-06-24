# preprocessing_info.py
"""
File: smartcash/ui/info_boxes/preprocessing_info.py
Deskripsi: Konten info box untuk konfigurasi preprocessing
"""

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.components import create_info_accordion
from smartcash.ui.utils.constants import ICONS

def get_preprocessing_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk konfigurasi preprocessing.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi konfigurasi preprocessing
    """
    TITLE = "Tentang Konfigurasi Preprocessing"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Modul ini digunakan untuk memproses dataset sebelum digunakan untuk training.</p>
    <p>Pastikan dataset sudah diunduh dan disimpan di direktori yang benar sebelum memulai preprocessing.</p>
    
    <h4>{ICONS.get('settings', '⚙️')} Konfigurasi Preprocessing</h4>
    <ul>
        <li><strong>Resize</strong>: Mengubah ukuran gambar menjadi ukuran yang konsisten.</li>
        <li><strong>Normalisasi</strong>: Menormalisasi nilai piksel ke rentang 0-1.</li>
        <li><strong>Augmentasi</strong>: Menambahkan variasi data dengan transformasi gambar.</li>
    </ul>
    
    <h4>{ICONS.get('info', 'ℹ️')} Rekomendasi</h4>
    <ul>
        <li>Gunakan augmentasi untuk meningkatkan variasi data training.</li>
        <li>Pastikan ukuran gambar sesuai dengan input model yang akan digunakan.</li>
        <li>Normalisasi membantu model untuk konvergen lebih cepat.</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)