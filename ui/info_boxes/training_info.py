"""
File: smartcash/ui/info_boxes/training_info.py
Deskripsi: Informasi bantuan untuk modul training model
"""

import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS, ICONS

def get_training_info(open_by_default=True):
    """
    Mendapatkan informasi bantuan untuk modul training model.
    
    Args:
        open_by_default: Apakah accordion terbuka secara default
    
    Returns:
        Widget accordion berisi informasi bantuan
    """
    # Informasi umum
    general_info = widgets.HTML(
        f"""
        <div style='padding: 10px;'>
            <p><strong>{ICONS.get('info', 'ℹ️')} Tentang Training Model</strong></p>
            <p>Modul ini digunakan untuk melatih model deteksi mata uang dengan konfigurasi yang telah diatur sebelumnya.</p>
            <p>Pastikan dataset telah dipreprocessing dan konfigurasi model telah diatur dengan benar sebelum memulai training.</p>
            
            <p><strong>{ICONS.get('settings', '⚙️')} Konfigurasi Training</strong></p>
            <ul>
                <li><strong>Backbone</strong>: Arsitektur backbone yang digunakan untuk model (EfficientNet-B4 direkomendasikan).</li>
                <li><strong>Epochs</strong>: Jumlah iterasi training pada seluruh dataset.</li>
                <li><strong>Batch Size</strong>: Jumlah sampel yang diproses dalam satu iterasi.</li>
                <li><strong>Learning Rate</strong>: Tingkat pembelajaran model, mempengaruhi kecepatan konvergensi.</li>
            </ul>
            
            <p><strong>{ICONS.get('check', '✓')} Opsi Training</strong></p>
            <ul>
                <li><strong>Simpan checkpoint</strong>: Menyimpan model pada interval tertentu selama training.</li>
                <li><strong>Gunakan TensorBoard</strong>: Mengaktifkan visualisasi metrik training dengan TensorBoard.</li>
                <li><strong>Gunakan Mixed Precision</strong>: Mempercepat training dengan menggunakan FP16 dan FP32.</li>
                <li><strong>Gunakan EMA</strong>: Menggunakan Exponential Moving Average untuk bobot model yang lebih stabil.</li>
            </ul>
        </div>
        """
    )
    
    # Informasi metrik
    metrics_info = widgets.HTML(
        f"""
        <div style='padding: 10px;'>
            <p><strong>{ICONS.get('chart', '📊')} Metrik Training</strong></p>
            <ul>
                <li><strong>Loss</strong>: Nilai error model, semakin rendah semakin baik.</li>
                <li><strong>mAP (mean Average Precision)</strong>: Metrik utama untuk evaluasi model deteksi objek.</li>
                <li><strong>Precision</strong>: Rasio prediksi positif yang benar terhadap total prediksi positif.</li>
                <li><strong>Recall</strong>: Rasio prediksi positif yang benar terhadap total positif sebenarnya.</li>
            </ul>
            
            <p><strong>{ICONS.get('warning', '⚠️')} Hal yang Perlu Diperhatikan</strong></p>
            <ul>
                <li>Nilai loss yang sangat rendah bisa mengindikasikan overfitting.</li>
                <li>Perhatikan gap antara training loss dan validation loss.</li>
                <li>mAP yang tinggi dengan recall rendah menunjukkan model hanya mendeteksi objek yang mudah.</li>
                <li>Perhatikan kurva learning untuk menentukan kapan harus menghentikan training.</li>
            </ul>
        </div>
        """
    )
    
    # Informasi troubleshooting
    troubleshooting_info = widgets.HTML(
        f"""
        <div style='padding: 10px;'>
            <p><strong>{ICONS.get('bug', '🐛')} Troubleshooting</strong></p>
            <ul>
                <li><strong>Out of Memory</strong>: Kurangi batch size atau gunakan mixed precision.</li>
                <li><strong>Loss menjadi NaN</strong>: Kurangi learning rate atau periksa kembali preprocessing dataset.</li>
                <li><strong>Performa rendah</strong>: Periksa kualitas dataset, coba augmentasi lebih agresif.</li>
                <li><strong>Training lambat</strong>: Aktifkan mixed precision, gunakan batch size yang optimal.</li>
            </ul>
            
            <p><strong>{ICONS.get('tip', '💡')} Tips</strong></p>
            <ul>
                <li>Mulai dengan learning rate yang kecil dan tingkatkan secara bertahap.</li>
                <li>Gunakan checkpoint untuk melanjutkan training jika terjadi interupsi.</li>
                <li>Evaluasi model secara berkala pada validation set.</li>
                <li>Perhatikan kurva learning untuk mendeteksi overfitting atau underfitting.</li>
            </ul>
        </div>
        """
    )
    
    # Buat accordion
    accordion = widgets.Accordion(
        children=[general_info, metrics_info, troubleshooting_info],
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Set judul accordion
    accordion.set_title(0, f"{ICONS.get('info', 'ℹ️')} Informasi Umum")
    accordion.set_title(1, f"{ICONS.get('chart', '📊')} Metrik Training")
    accordion.set_title(2, f"{ICONS.get('bug', '🐛')} Troubleshooting & Tips")
    
    # Set accordion yang terbuka secara default
    accordion.selected_index = 0 if open_by_default else None
    
    return accordion
