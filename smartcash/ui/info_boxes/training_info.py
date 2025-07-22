# training_info.py
"""
File: smartcash/ui/info_boxes/training_info.py
Deskripsi: Konten info box untuk konfigurasi training model
"""

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.components import create_info_accordion
from smartcash.ui.utils.constants import ICONS

def get_training_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk konfigurasi training model.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi konfigurasi training
    """
    TITLE = "Tentang Konfigurasi Training Model"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Modul ini digunakan untuk melatih model deteksi mata uang dengan konfigurasi yang telah diatur sebelumnya.</p>
    <p>Pastikan dataset telah dipreprocessing dan konfigurasi model telah diatur dengan benar sebelum memulai training.</p>
    
    <h4>{ICONS.get('settings', '⚙️')} Konfigurasi Training</h4>
    <ul>
        <li><strong>Backbone</strong>: Arsitektur backbone yang digunakan untuk model (EfficientNet-B4 direkomendasikan).</li>
        <li><strong>Epochs</strong>: Jumlah iterasi training pada seluruh dataset.</li>
        <li><strong>Batch Size</strong>: Jumlah sampel yang diproses dalam satu iterasi.</li>
        <li><strong>Learning Rate</strong>: Tingkat pembelajaran model, mempengaruhi kecepatan konvergensi.</li>
    </ul>
    
    <h4>{ICONS.get('check', '✓')} Opsi Training</h4>
    <ul>
        <li><strong>Simpan checkpoint</strong>: Menyimpan model pada interval tertentu selama training.</li>
        <li><strong>Validasi</strong>: Mengevaluasi model pada validasi set setelah setiap epoch.</li>
        <li><strong>Early Stopping</strong>: Menghentikan training jika performa model tidak membaik.</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)