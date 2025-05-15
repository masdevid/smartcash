"""
File: smartcash/ui/info_boxes/training_strategy_info.py
Deskripsi: Konten info box untuk konfigurasi strategi pelatihan
"""

import ipywidgets as widgets
from typing import Dict, Any

def get_training_strategy_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk konfigurasi strategi pelatihan.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi strategi pelatihan
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Tentang Konfigurasi Strategi Pelatihan"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Strategi pelatihan mengontrol bagaimana model dilatih, divalidasi, dan dimonitor selama proses training. Pengaturan yang tepat dapat meningkatkan efisiensi training dan kualitas model akhir.</p>
    
    <h4>Parameter Utilitas</h4>
    <ul>
        <li><strong>Experiment Name</strong>: Nama eksperimen yang digunakan untuk pelacakan dan penamaan model. Gunakan nama yang deskriptif untuk memudahkan identifikasi model.</li>
        <li><strong>Checkpoint Dir</strong>: Direktori tempat model checkpoint akan disimpan. Pastikan direktori ini mudah diakses dan memiliki ruang penyimpanan yang cukup.</li>
        <li><strong>TensorBoard</strong>: Mengaktifkan visualisasi metrik training secara real-time. Sangat berguna untuk memantau proses training dan mendiagnosis masalah.</li>
        <li><strong>Log Metrics Every</strong>: Frekuensi pencatatan metrik dalam batch. Nilai yang lebih kecil memberikan monitoring yang lebih detail tetapi dapat memperlambat training.</li>
        <li><strong>Visualize Batch Every</strong>: Frekuensi visualisasi batch dalam batch. Membantu memahami bagaimana model memproses gambar selama training.</li>
        <li><strong>Gradient Clipping</strong>: Membatasi nilai gradien untuk mencegah exploding gradients. Nilai antara 0.5-1.0 biasanya efektif untuk model deteksi objek.</li>
        <li><strong>Mixed Precision</strong>: Menggunakan presisi campuran (FP16/FP32) untuk mempercepat training dan mengurangi penggunaan memori. Direkomendasikan untuk GPU yang mendukung.</li>
        <li><strong>Layer Mode</strong>: Strategi untuk membekukan atau melatih layer model.
            <ul>
                <li><em>Train All</em>: Melatih semua layer model, memberikan fleksibilitas maksimum tetapi membutuhkan lebih banyak data dan waktu.</li>
                <li><em>Freeze Backbone</em>: Membekukan layer backbone, cocok untuk dataset kecil atau saat menggunakan pre-trained model.</li>
                <li><em>Freeze Partial</em>: Membekukan sebagian layer, menyeimbangkan antara fleksibilitas dan efisiensi.</li>
            </ul>
        </li>
    </ul>
    
    <h4>Parameter Validasi</h4>
    <ul>
        <li><strong>Validation Frequency</strong>: Seberapa sering model dievaluasi pada dataset validasi. Evaluasi yang lebih sering memberikan monitoring yang lebih baik tetapi memperlambat training.</li>
        <li><strong>IoU Threshold</strong>: Threshold Intersection over Union untuk menentukan deteksi yang benar. Nilai yang lebih tinggi membutuhkan lokalisasi yang lebih tepat.</li>
        <li><strong>Confidence Threshold</strong>: Threshold kepercayaan minimum untuk deteksi. Nilai yang lebih tinggi mengurangi false positives tetapi dapat meningkatkan false negatives.</li>
    </ul>
    
    <h4>Multi-scale Training</h4>
    <ul>
        <li><strong>Enable Multi-scale</strong>: Melatih model dengan ukuran gambar yang bervariasi. Meningkatkan ketahanan model terhadap variasi ukuran objek dalam gambar nyata.</li>
        <li><strong>Scale Range</strong>: Rentang faktor skala yang digunakan. Rentang yang lebih lebar memberikan variasi yang lebih besar tetapi dapat memperlambat training.</li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)

def get_utils_training_strategy_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter utilitas training.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter utilitas
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Parameter Utilitas Training"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <h4>Pengaturan Eksperimen</h4>
    <ul>
        <li><strong>Experiment Name</strong>: Nama untuk mengidentifikasi eksperimen.
            <ul>
                <li>Gunakan nama yang deskriptif dan unik untuk setiap eksperimen.</li>
                <li>Contoh: "EfficientNetB4_YOLOv5s_Rupiah_v1"</li>
            </ul>
        </li>
        <li><strong>Checkpoint Dir</strong>: Direktori untuk menyimpan model.
            <ul>
                <li>Pastikan direktori mudah diakses dan memiliki ruang yang cukup.</li>
                <li>Untuk Google Colab, gunakan path yang terhubung ke Google Drive.</li>
            </ul>
        </li>
    </ul>
    
    <h4>Monitoring</h4>
    <ul>
        <li><strong>TensorBoard</strong>: Visualisasi metrik secara real-time.
            <ul>
                <li>Sangat berguna untuk memantau loss, akurasi, dan metrik lainnya.</li>
                <li>Aktifkan untuk eksperimen penting atau saat melakukan tuning hyperparameter.</li>
            </ul>
        </li>
        <li><strong>Log Metrics Every</strong>: Frekuensi pencatatan metrik.
            <ul>
                <li>Nilai kecil (10-50): Monitoring detail, berguna untuk debugging.</li>
                <li>Nilai sedang (100-200): Keseimbangan antara detail dan kinerja.</li>
                <li>Nilai besar (>500): Minimal overhead, cocok untuk training lama.</li>
            </ul>
        </li>
        <li><strong>Visualize Batch Every</strong>: Frekuensi visualisasi batch.
            <ul>
                <li>Membantu memahami bagaimana model memproses gambar.</li>
                <li>Nilai yang lebih besar mengurangi overhead training.</li>
            </ul>
        </li>
    </ul>
    
    <h4>Optimasi Kinerja</h4>
    <ul>
        <li><strong>Gradient Clipping</strong>: Membatasi nilai gradien.
            <ul>
                <li>Mencegah exploding gradients dan meningkatkan stabilitas training.</li>
                <li>Nilai 0.5-1.0 biasanya efektif untuk model deteksi objek.</li>
            </ul>
        </li>
        <li><strong>Mixed Precision</strong>: Menggunakan presisi campuran.
            <ul>
                <li>Dapat mempercepat training hingga 2-3x pada GPU yang mendukung.</li>
                <li>Mengurangi penggunaan memori, memungkinkan batch size yang lebih besar.</li>
            </ul>
        </li>
        <li><strong>Layer Mode</strong>: Strategi untuk membekukan atau melatih layer.
            <ul>
                <li><em>Train All</em>: Fleksibilitas maksimum, membutuhkan lebih banyak data.</li>
                <li><em>Freeze Backbone</em>: Cocok untuk dataset kecil, mempertahankan fitur pre-trained.</li>
                <li><em>Freeze Partial</em>: Keseimbangan antara adaptasi dan stabilitas.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)

def get_validation_training_strategy_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter validasi.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter validasi
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Parameter Validasi"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <h4>Pengaturan Validasi</h4>
    <ul>
        <li><strong>Validation Frequency</strong>: Seberapa sering model dievaluasi.
            <ul>
                <li>Nilai kecil (1-2): Evaluasi lebih sering, monitoring lebih detail.</li>
                <li>Nilai sedang (3-5): Keseimbangan antara monitoring dan kecepatan.</li>
                <li>Nilai besar (>10): Minimal overhead, cocok untuk dataset besar.</li>
                <li>Untuk deteksi mata uang, nilai 1-3 biasanya memberikan monitoring yang cukup.</li>
            </ul>
        </li>
        <li><strong>IoU Threshold</strong>: Threshold untuk menentukan deteksi yang benar.
            <ul>
                <li>Nilai 0.5: Standar untuk kebanyakan tugas deteksi objek.</li>
                <li>Nilai lebih tinggi (0.6-0.7): Membutuhkan lokalisasi yang lebih tepat.</li>
                <li>Untuk deteksi mata uang, nilai 0.5-0.6 biasanya sesuai.</li>
            </ul>
        </li>
        <li><strong>Confidence Threshold</strong>: Threshold kepercayaan minimum.
            <ul>
                <li>Nilai rendah (0.1-0.3): Lebih banyak deteksi, termasuk yang kurang yakin.</li>
                <li>Nilai sedang (0.3-0.5): Keseimbangan antara recall dan precision.</li>
                <li>Nilai tinggi (>0.5): Hanya deteksi dengan kepercayaan tinggi.</li>
                <li>Untuk deteksi mata uang, nilai 0.3-0.4 biasanya memberikan keseimbangan yang baik.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)

def get_multiscale_training_strategy_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter multi-scale.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter multi-scale
    """
    from smartcash.ui.utils.info_utils import create_info_accordion
    from smartcash.ui.utils.constants import ICONS
    
    TITLE = "Parameter Multi-scale"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <h4>Multi-scale Training</h4>
    <ul>
        <li><strong>Enable Multi-scale</strong>: Melatih dengan ukuran gambar bervariasi.
            <ul>
                <li>Meningkatkan ketahanan model terhadap variasi ukuran objek.</li>
                <li>Sangat berguna untuk deteksi mata uang yang dapat muncul dalam berbagai ukuran.</li>
                <li>Dapat memperlambat training tetapi biasanya memberikan model yang lebih robust.</li>
            </ul>
        </li>
        <li><strong>Scale Range</strong>: Rentang faktor skala yang digunakan.
            <ul>
                <li>Rentang kecil (0.8-1.2): Variasi minimal, overhead minimal.</li>
                <li>Rentang sedang (0.5-1.5): Keseimbangan antara variasi dan kinerja.</li>
                <li>Rentang besar (0.3-1.7): Variasi maksimal, cocok untuk kasus dengan variasi ukuran ekstrem.</li>
                <li>Untuk deteksi mata uang, rentang 0.5-1.5 biasanya memberikan variasi yang cukup.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)
