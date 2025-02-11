# SmartCash Detector: Optimasi Deteksi Nominal Mata Uang dengan YOLOv5 dan EfficientNet-B4

## 🎯 Tujuan Penelitian

Proyek SmartCash Detector bertujuan mengembangkan sistem deteksi mata uang Rupiah yang akurat dan andal menggunakan teknologi deteksi objek YOLOv5. Fokus utama penelitian adalah mengoptimalkan algoritma YOLOv5 DarkNet dengan arsitektur backbone EfficientNet-B4 untuk meningkatkan performa deteksi, terutama pada kondisi spesifik.

## 🔬 Fitur Utama

- 🤖 Integrasi YOLOv5 dengan EfficientNet-B4
- 📷 Deteksi nominal mata uang Rupiah
- 🌈 Robust terhadap variasi pencahayaan
- 🔍 Optimasi deteksi objek kecil
- 📊 Evaluasi komprehensif dengan berbagai skenario

## 🛠 Persyaratan Sistem

### Perangkat Keras
- GPU CUDA dengan memori minimal 8GB
- CPU Intel/AMD dengan dukunng AVX2
- RAM minimal 16GB

### Perangkat Lunak
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+
- Albumentations
- EfficientNet-PyTorch

## 🚀 Instalasi

```bash
# Clone repository
git clone https://github.com/alfridasabar/smartcash-detector.git
cd smartcash-detector

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📋 Penggunaan Dasar

### Persiapan Dataset
```bash
python src/data/prepare.py
```

### Training Model
```bash
python src/main.py
# Pilih menu Training Model
```

### Evaluasi
```bash
python src/main.py
# Pilih menu Evaluasi & Pengujian
```

## 🧪 Skenario Pengujian

1. **Pencahayaan Normal**
   - Deteksi mata uang dalam kondisi cahaya standar

2. **Pencahayaan Rendah**
   - Uji kemampuan deteksi dalam kondisi cahaya minim

3. **Deteksi Objek Kecil**
   - Fokus pada akurasi deteksi nominal berukuran kecil

4. **Oklusi Parsial**
   - Simulasi deteksi mata uang yang terhalang sebagian

## 📈 Metrik Evaluasi

- mAP (Mean Average Precision)
- Precision
- Recall
- Waktu Inferensi
- Performa Deteksi Objek Kecil

## 🤝 Kontribusi

Kami mendorong kontribusi dari komunitas! Silakan buka _issue_ atau kirim _pull request_.

### Panduan Kontribusi
1. Fork repository
2. Buat branch fitur (`git checkout -b fitur/AturDeteksi`)
3. Commit perubahan (`git commit -m 'Tambah fitur deteksi lanjutan'`)
4. Push ke branch (`git push origin fitur/AturDeteksi`)
5. Buka Pull Request

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.

## 📞 Kontak

**Alfrida Sabar**
- Email: alfrida.sabar@gmail.com
- LinkedIn: [Profil LinkedIn](https://www.linkedin.com/in/@username/)
- ResearchGate: [Profil ResearchGate](https://www.researchgate.net/profile/@username)

## 🏆 Sitasi

Jika penelitian ini membantu Anda, pertimbangkan untuk mensitasi:

```
Sabar, A. (2025). Optimasi Deteksi Nominal Mata Uang dengan YOLOv5 dan EfficientNet-B4. (Unpublished)
```