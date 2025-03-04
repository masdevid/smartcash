# Panduan Anotasi Dataset SmartCash di Roboflow

## Informasi Dataset Utama

### Workspace
- **Nama**: `smartcash-wo2us`
- **Deskripsi**: Workspace untuk dataset deteksi mata uang Rupiah

### Projekt Details
- **Nama Projekt**: `rupiah-emisi-2022`
- **Versi**: `3`
- **Lisensi**: CC BY 4.0

## Struktur Kelas
Total 17 kelas yang dibagi dalam 3 layer:

#### Layer 1 (Banknote)
1. `001` - 1.000 Rupiah
2. `002` - 2.000 Rupiah
3. `005` - 5.000 Rupiah
4. `010` - 10.000 Rupiah
5. `020` - 20.000 Rupiah
6. `050` - 50.000 Rupiah
7. `100` - 100.000 Rupiah

#### Layer 2 (Nominal Area)
8. `l2_001` - Area nominal 1.000
9. `l2_002` - Area nominal 2.000
10. `l2_005` - Area nominal 5.000
11. `l2_010` - Area nominal 10.000
12. `l2_020` - Area nominal 20.000
13. `l2_050` - Area nominal 50.000
14. `l2_100` - Area nominal 100.000

#### Layer 3 (Security Features)
15. `l3_sign` - Tanda keamanan
16. `l3_text` - Teks keamanan
17. `l3_thread` - Security thread

## Aturan Bounding Box
- Bounding box harus mencakup seluruh bagian uang kertas
- Margin: berikan margin 2-3 pixel dari tepi uang
- Pastikan bounding box tetap rapat meski uang dalam kondisi miring
- Untuk uang terlipat: anotasi mengikuti bentuk terlihat

## Anotasi Berdasarkan Kondisi

### a. Variasi Posisi
- Posisi Landscape: anotasi normal
- Posisi Portrait: anotasi mencakup seluruh tinggi
- Posisi Miring: anotasi mengikuti orientasi
- Terlipat Sebagian: anotasi bagian yang terlihat
- Multiple Uang: anotasi terpisah untuk setiap uang

### b. Variasi Pencahayaan
- Pencahayaan Normal: anotasi standard
- Pencahayaan Redup: pastikan bounding box tepat di tepi
- Pencahayaan Sangat Terang: ikuti outline yang masih terlihat
- Bayangan: jangan masukkan area bayangan dalam box

## Preprocessing

### Transformasi Default
- Resize: 640x640 piksel
- Normalisasi:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

## Quality Control
- Validasi Minimum per Gambar:
  1. Label kelas sesuai nominal
  2. Bounding box mencakup seluruh uang
  3. Tidak ada area kosong berlebih
  4. Konsistensi pada gambar sejenis

## Penanganan Kasus Khusus
- Uang Terlipat > 50%: tetap anotasi jika nominal terbaca
- Tertutup Sebagian: anotasi jika minimal 75% terlihat
- Blur/Tidak Fokus: anotasi jika nominal terbaca jelas
- Multiple Overlap: anotasi terpisah selama terlihat >75%

## Tips Efisiensi
1. Gunakan shortcut keyboard Roboflow
2. Mulai dari gambar pencahayaan normal
3. Validasi per batch (50-100 gambar)
4. Catat kasus unik untuk evaluasi

## Integrasi API

### Metode Download
- REST API download
- Progress bar tracking
- Validasi struktur dataset
- Ekstraksi otomatis

### Konfigurasi API
```python
roboflow_config = {
    'workspace': 'smartcash-wo2us',
    'project': 'rupiah-emisi-2022',
    'version': '3',
    'api_key': 'your_roboflow_api_key'
}
```

## Protokol Validasi Dataset

### Validasi Struktur
- Keberadaan direktori `images` dan `labels`
- Jumlah gambar dan label sesuai
- Format file yang konsisten

### Penanganan Multi-Layer
- Dukungan anotasi parsial
- Fleksibel untuk gambar dengan subset annotations
- Pemrosesan kondisional berdasarkan layer aktif

## Catatan Penting
- Pastikan API key tersedia
- Gunakan environment variable/secrets untuk menyimpan API key
- Selalu validasi dataset setelah download

## Referensi
- URL Dataset: [Roboflow Universe Link](https://universe.roboflow.com/smartcash-wo2us/rupiah-emisi-2022)