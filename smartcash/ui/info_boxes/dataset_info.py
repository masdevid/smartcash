"""
File: smartcash/ui/info_boxes/dataset_info.py
Deskripsi: Konten info box panduan dataset
"""

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.components import create_info_accordion
from smartcash.ui.utils.constants import ICONS

def get_dataset_info(open_by_default: bool = False) -> widgets.Accordion:
    TITLE = "Panduan Dataset"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <h4 style="color:inherit">📥 Endpoint Dataset</h4>
    <p>Ada tiga sumber dataset yang dapat digunakan:</p>
    <ul>
        <li><strong>Roboflow</strong>: Memerlukan API key dan detail project.</li>
        <li><strong>Google Drive</strong>: Mengambil dataset dari folder Drive (hanya di Colab).</li>
        <li><strong>URL Kustom</strong>: Download dari URL file ZIP atau arsip lainnya.</li>
    </ul>
    
    <h4 style="color:inherit">🏷️ Format Dataset</h4>
    <p>Pilih format output sesuai kebutuhan training:</p>
    <ul>
        <li><strong>YOLO v5</strong>: Format dataset standar untuk YOLOv5.</li>
        <li><strong>COCO</strong>: Format JSON yang kompatibel dengan MS COCO.</li>
        <li><strong>VOC</strong>: Format Pascal VOC (XML).</li>
    </ul>
    
    <h4 style="color:inherit">🧠 Struktur Dataset untuk YOLOv5</h4>
    <p>Dataset harus memiliki struktur direktori sebagai berikut:</p>
    <pre style="background:#f5f5f5; padding:10px; border-radius:5px;">
    data/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img1.txt
    │       └── ...
    ├── valid/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── img1.txt
    │       └── ...
    └── test/
        ├── images/
        │   ├── img1.jpg
        │   └── ...
        └── labels/
            ├── img1.txt
            └── ...
    </pre>
    
    <h4 style="color:inherit">📝 Format Label YOLOv5</h4>
    <p>Label YOLOv5 menggunakan format berikut (satu objek per baris):</p>
    <pre style="background:#f5f5f5; padding:10px; border-radius:5px;">
    &lt;class_id&gt; &lt;x_center&gt; &lt;y_center&gt; &lt;width&gt; &lt;height&gt;
    </pre>
    <p>Semua nilai koordinat dinormalisasi ke rentang 0-1 relatif terhadap ukuran gambar.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", f"{ICONS['data']}", open_by_default)