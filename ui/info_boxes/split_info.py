"""
File: smartcash/ui/info_boxes/split_info.py
Deskripsi: Konten info box untuk konfigurasi split dataset
"""

import ipywidgets as widgets

def get_split_info(open_by_default: bool = False) -> widgets.Accordion:
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS

    TITLE = "Tentang Konfigurasi Split Dataset"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Pembagian dataset menjadi 3 subset:</p>
    <ul>
        <li><strong>Train</strong>: Dataset untuk pelatihan model (biasanya 70-80%)</li>
        <li><strong>Validation</strong>: Dataset untuk validasi selama pelatihan (biasanya 10-15%)</li>
        <li><strong>Test</strong>: Dataset untuk evaluasi akhir model (biasanya 10-15%)</li>
    </ul>
    <p>Gunakan <strong>stratified split</strong> untuk memastikan distribusi kelas tetap seimbang di semua subset.</p>
    
    <h4 style="color:inherit">{ICONS['folder']} Lokasi Dataset</h4>
    <p>Data mentah dan data terpreprocessing akan diambil dari lokasi yang dikonfigurasi:</p>
    <ul>
        <li>Dataset mentah: <code>/content/drive/MyDrive/SmartCash/data</code></li>
        <li>Dataset preprocessed: <code>/content/drive/MyDrive/SmartCash/data/preprocessed</code></li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", "⚙️", open_by_default)
