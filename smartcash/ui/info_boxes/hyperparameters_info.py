"""
File: smartcash/ui/info_boxes/hyperparameters_info.py
Deskripsi: Konten info box untuk konfigurasi hyperparameter
"""

# Third-party imports
import ipywidgets as widgets

# Local application imports
from smartcash.ui.components import create_info_accordion
from smartcash.ui.utils.constants import ICONS

def get_hyperparameters_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk konfigurasi hyperparameter.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi hyperparameter
    """
    TITLE = "Tentang Konfigurasi Hyperparameter"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <p>Hyperparameter adalah parameter yang mengontrol proses training model. Pengaturan yang tepat dapat meningkatkan akurasi dan performa model.</p>
    
    <h4>Parameter Dasar</h4>
    <ul>
        <li><strong>Batch Size</strong>: Jumlah gambar yang diproses dalam satu iterasi. Nilai yang lebih besar dapat mempercepat training tetapi membutuhkan lebih banyak memori GPU. Nilai yang lebih kecil dapat meningkatkan generalisasi model.</li>
        <li><strong>Image Size</strong>: Ukuran gambar input dalam piksel. Ukuran yang lebih besar dapat meningkatkan akurasi tetapi membutuhkan lebih banyak memori dan waktu komputasi.</li>
        <li><strong>Epochs</strong>: Jumlah kali model akan melihat seluruh dataset training. Terlalu sedikit epoch dapat menyebabkan underfitting, terlalu banyak dapat menyebabkan overfitting.</li>
        <li><strong>Augmentasi</strong>: Teknik untuk meningkatkan variasi data training dengan melakukan transformasi seperti rotasi, flip, dan perubahan kecerahan. Membantu model lebih robust dan mengurangi overfitting.</li>
    </ul>
    
    <h4>Parameter Optimasi</h4>
    <ul>
        <li><strong>Optimizer</strong>: Algoritma yang digunakan untuk memperbarui bobot model selama training.
            <ul>
                <li><em>SGD</em>: Stochastic Gradient Descent, algoritma klasik yang dapat memberikan generalisasi yang baik.</li>
                <li><em>Adam</em>: Adaptive Moment Estimation, biasanya konvergen lebih cepat dari SGD dan bekerja baik untuk banyak kasus.</li>
                <li><em>AdamW</em>: Varian Adam dengan weight decay yang lebih baik, sering memberikan hasil yang lebih baik untuk model besar.</li>
                <li><em>RMSprop</em>: Root Mean Square Propagation, baik untuk masalah non-stasioner dan RNN.</li>
            </ul>
        </li>
        <li><strong>Learning Rate</strong>: Ukuran langkah yang diambil dalam proses optimasi. Nilai yang terlalu besar dapat menyebabkan divergensi, nilai yang terlalu kecil dapat menyebabkan konvergensi yang lambat atau terjebak di local minima.</li>
        <li><strong>Momentum</strong>: Membantu optimizer mengatasi local minima dan mempercepat konvergensi dengan mempertahankan arah update sebelumnya.</li>
        <li><strong>Weight Decay</strong>: Teknik regularisasi yang mengurangi kompleksitas model dengan menambahkan penalti pada bobot yang besar, membantu mencegah overfitting.</li>
    </ul>
    
    <h4>Learning Rate Scheduler</h4>
    <ul>
        <li><strong>Scheduler</strong>: Strategi untuk mengubah learning rate selama training.
            <ul>
                <li><em>None</em>: Learning rate tetap konstan sepanjang training.</li>
                <li><em>Cosine</em>: Learning rate menurun mengikuti kurva cosinus, memberikan penurunan yang halus.</li>
                <li><em>Linear</em>: Learning rate menurun secara linear dari nilai awal ke nilai minimum.</li>
                <li><em>Step</em>: Learning rate dikurangi dengan faktor tertentu pada epoch tertentu.</li>
            </ul>
        </li>
        <li><strong>Warmup Epochs</strong>: Jumlah epoch di awal training di mana learning rate meningkat secara bertahap. Membantu stabilitas training di awal.</li>
        <li><strong>Warmup Momentum</strong>: Nilai momentum selama fase warmup.</li>
        <li><strong>Warmup Bias LR</strong>: Learning rate khusus untuk bias selama fase warmup.</li>
    </ul>
    
    <h4>Parameter Lanjutan</h4>
    <ul>
        <li><strong>Early Stopping</strong>: Menghentikan training jika tidak ada peningkatan performa setelah beberapa epoch, mencegah overfitting dan menghemat waktu komputasi.
            <ul>
                <li><em>Patience</em>: Jumlah epoch untuk menunggu sebelum menghentikan training jika tidak ada peningkatan.</li>
                <li><em>Min Delta</em>: Perubahan minimum yang dianggap sebagai peningkatan.</li>
            </ul>
        </li>
        <li><strong>Checkpoint</strong>: Menyimpan model terbaik selama training.
            <ul>
                <li><em>Simpan Model Terbaik</em>: Menyimpan model dengan performa terbaik pada metrik yang dipilih.</li>
                <li><em>Metric</em>: Metrik yang digunakan untuk menentukan model terbaik, seperti mAP (mean Average Precision).</li>
            </ul>
        </li>
    </ul>
    
    <p><strong>Catatan</strong>: Kombinasi hyperparameter yang optimal dapat bervariasi tergantung pada dataset dan tugas spesifik. Eksperimen dengan berbagai konfigurasi dapat membantu menemukan pengaturan terbaik untuk kasus Anda.</p>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('settings', '⚙️'), open_by_default)

def get_basic_hyperparameters_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter dasar.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter dasar
    """
    TITLE = "Parameter Dasar"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <ul>
        <li><strong>Batch Size</strong>: Jumlah gambar yang diproses dalam satu iterasi. 
            <ul>
                <li>Nilai yang lebih besar (32-64) dapat mempercepat training tetapi membutuhkan lebih banyak memori GPU.</li>
                <li>Nilai yang lebih kecil (8-16) dapat meningkatkan generalisasi model dan bekerja dengan GPU yang memiliki memori terbatas.</li>
                <li>Untuk deteksi mata uang, batch size 16-32 biasanya memberikan keseimbangan yang baik.</li>
            </ul>
        </li>
        <li><strong>Image Size</strong>: Ukuran gambar input dalam piksel.
            <ul>
                <li>Ukuran yang lebih besar (640-1280) dapat meningkatkan akurasi untuk objek kecil tetapi membutuhkan lebih banyak memori dan waktu komputasi.</li>
                <li>Ukuran yang lebih kecil (320-512) lebih cepat tetapi mungkin kehilangan detail penting.</li>
                <li>Untuk deteksi mata uang, ukuran 640x640 biasanya memberikan keseimbangan yang baik.</li>
            </ul>
        </li>
        <li><strong>Epochs</strong>: Jumlah kali model akan melihat seluruh dataset training.
            <ul>
                <li>Terlalu sedikit epoch (<50) dapat menyebabkan underfitting (model belum belajar pola yang cukup).</li>
                <li>Terlalu banyak epoch (>200) dapat menyebabkan overfitting (model menghafal data training).</li>
                <li>Untuk deteksi mata uang dengan transfer learning, 100-150 epoch biasanya cukup.</li>
            </ul>
        </li>
        <li><strong>Augmentasi</strong>: Teknik untuk meningkatkan variasi data training.
            <ul>
                <li>Sangat membantu ketika dataset terbatas.</li>
                <li>Meningkatkan ketahanan model terhadap variasi dalam kondisi nyata (pencahayaan, sudut, dll).</li>
                <li>Untuk deteksi mata uang, augmentasi sangat direkomendasikan karena variasi kondisi pengambilan gambar.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)

def get_optimization_hyperparameters_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter optimasi.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter optimasi
    """
    TITLE = "Parameter Optimasi"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    <ul>
        <li><strong>Optimizer</strong>: Algoritma yang digunakan untuk memperbarui bobot model.
            <ul>
                <li><em>SGD</em>: Memberikan generalisasi yang baik, tetapi konvergensi lebih lambat. Baik untuk fine-tuning.</li>
                <li><em>Adam</em>: Konvergen lebih cepat, baik untuk training dari awal. Dapat memberikan hasil yang baik dengan sedikit tuning.</li>
                <li><em>AdamW</em>: Varian Adam dengan weight decay yang lebih baik, sering memberikan hasil yang lebih baik untuk model besar.</li>
                <li><em>RMSprop</em>: Alternatif yang baik ketika Adam tidak memberikan hasil yang optimal.</li>
                <li>Untuk EfficientNet-B4 backbone, SGD atau AdamW biasanya memberikan hasil terbaik.</li>
            </ul>
        </li>
        <li><strong>Learning Rate</strong>: Ukuran langkah dalam proses optimasi.
            <ul>
                <li>Nilai yang terlalu besar (>0.1) dapat menyebabkan divergensi atau osilasi.</li>
                <li>Nilai yang terlalu kecil (<0.0001) dapat menyebabkan konvergensi yang sangat lambat.</li>
                <li>Untuk transfer learning dengan EfficientNet-B4, nilai 0.001-0.01 biasanya optimal.</li>
                <li>Untuk training dari awal, nilai yang lebih besar (0.01-0.1) mungkin diperlukan.</li>
            </ul>
        </li>
        <li><strong>Momentum</strong>: Membantu optimizer mengatasi local minima.
            <ul>
                <li>Nilai yang umum adalah 0.9-0.99.</li>
                <li>Nilai yang lebih tinggi (0.95-0.99) dapat membantu konvergensi tetapi mungkin membuat training kurang stabil.</li>
                <li>Untuk SGD, momentum sangat penting dan nilai 0.937 biasanya memberikan hasil yang baik.</li>
            </ul>
        </li>
        <li><strong>Weight Decay</strong>: Teknik regularisasi untuk mencegah overfitting.
            <ul>
                <li>Nilai yang umum adalah 0.0001-0.001.</li>
                <li>Nilai yang lebih tinggi memberikan regularisasi yang lebih kuat tetapi dapat menghambat pembelajaran.</li>
                <li>Untuk deteksi mata uang, nilai 0.0005 biasanya memberikan keseimbangan yang baik.</li>
            </ul>
        </li>
    </ul>
    
    <h4>Learning Rate Scheduler</h4>
    <ul>
        <li><strong>Scheduler</strong>: Strategi untuk mengubah learning rate.
            <ul>
                <li><em>None</em>: Gunakan jika dataset kecil atau epoch sedikit.</li>
                <li><em>Cosine</em>: Pilihan terbaik untuk kebanyakan kasus, memberikan penurunan yang halus.</li>
                <li><em>Linear</em>: Alternatif yang baik untuk cosine, lebih agresif di awal.</li>
                <li><em>Step</em>: Baik jika Anda tahu kapan harus menurunkan learning rate.</li>
            </ul>
        </li>
        <li><strong>Warmup Epochs</strong>: Meningkatkan learning rate secara bertahap di awal.
            <ul>
                <li>Membantu stabilitas training, terutama dengan batch size besar.</li>
                <li>Nilai 3-5 epoch biasanya cukup.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)

def get_advanced_hyperparameters_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Mendapatkan info box untuk parameter lanjutan.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
        
    Returns:
        Accordion widget berisi informasi parameter lanjutan
    """
    TITLE = "Parameter Lanjutan"
    content = f"""
    <h3 style="margin-top:0; color:inherit">{TITLE}</h3>
    
    <h4>Early Stopping</h4>
    <ul>
        <li><strong>Aktifkan Early Stopping</strong>: Menghentikan training jika tidak ada peningkatan.
            <ul>
                <li>Menghemat waktu komputasi dan mencegah overfitting.</li>
                <li>Sangat direkomendasikan untuk kebanyakan kasus.</li>
            </ul>
        </li>
        <li><strong>Patience</strong>: Jumlah epoch untuk menunggu sebelum berhenti.
            <ul>
                <li>Nilai yang terlalu kecil (1-5) dapat menghentikan training terlalu dini.</li>
                <li>Nilai yang terlalu besar (>20) dapat membuat early stopping tidak efektif.</li>
                <li>Untuk deteksi mata uang, nilai 10-15 biasanya memberikan keseimbangan yang baik.</li>
            </ul>
        </li>
        <li><strong>Min Delta</strong>: Perubahan minimum yang dianggap sebagai peningkatan.
            <ul>
                <li>Nilai yang lebih kecil (0.0001-0.001) lebih sensitif terhadap peningkatan kecil.</li>
                <li>Nilai yang lebih besar (0.01-0.05) hanya mempertimbangkan peningkatan yang signifikan.</li>
                <li>Untuk metrik mAP, nilai 0.001-0.005 biasanya sesuai.</li>
            </ul>
        </li>
    </ul>
    
    <h4>Checkpoint</h4>
    <ul>
        <li><strong>Simpan Model Terbaik</strong>: Menyimpan model dengan performa terbaik.
            <ul>
                <li>Sangat direkomendasikan untuk memastikan Anda mendapatkan model terbaik.</li>
                <li>Memungkinkan Anda untuk melanjutkan training dari checkpoint terbaik jika diperlukan.</li>
            </ul>
        </li>
        <li><strong>Metric</strong>: Metrik yang digunakan untuk menentukan model terbaik.
            <ul>
                <li><em>mAP_0.5</em>: Mean Average Precision dengan IoU threshold 0.5, metrik standar untuk deteksi objek.</li>
                <li><em>mAP_0.5:0.95</em>: Rata-rata mAP dengan IoU dari 0.5 hingga 0.95, lebih ketat dan komprehensif.</li>
                <li><em>precision</em>: Mengutamakan mengurangi false positives, baik jika false positives sangat merugikan.</li>
                <li><em>recall</em>: Mengutamakan mengurangi false negatives, baik jika false negatives sangat merugikan.</li>
                <li><em>f1</em>: Keseimbangan antara precision dan recall.</li>
                <li><em>loss</em>: Menggunakan fungsi loss langsung, dapat berguna untuk training awal.</li>
                <li>Untuk deteksi mata uang, mAP_0.5 atau mAP_0.5:0.95 biasanya pilihan terbaik.</li>
            </ul>
        </li>
    </ul>
    """
    
    return create_info_accordion(TITLE, content, "info", ICONS.get('info', 'ℹ️'), open_by_default)
