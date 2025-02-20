# Panduan Anotasi Dataset SmartCash di Roboflow

## 1. Struktur Kelas
- Format label: `{nominal}k` (contoh: `100k`, `50k`, dst)
- Total 7 kelas denominasi:
  - 100k - Seratus Ribu Rupiah
  - 50k  - Lima Puluh Ribu Rupiah
  - 20k  - Dua Puluh Ribu Rupiah
  - 10k  - Sepuluh Ribu Rupiah
  - 5k   - Lima Ribu Rupiah
  - 2k   - Dua Ribu Rupiah
  - 1k   - Seribu Rupiah

## 2. Aturan Bounding Box
- Bounding box harus mencakup seluruh bagian uang kertas
- Margin: berikan margin 2-3 pixel dari tepi uang
- Pastikan bounding box tetap rapat meski uang dalam kondisi miring
- Untuk uang terlipat: anotasi mengikuti bentuk terlihat

## 3. Anotasi Berdasarkan Kondisi

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

## 4. Quality Control
- Validasi Minimum per Gambar:
  1. Label kelas sesuai nominal
  2. Bounding box mencakup seluruh uang
  3. Tidak ada area kosong berlebih
  4. Konsistensi pada gambar sejenis

## 5. Penanganan Kasus Khusus
- Uang Terlipat > 50%: tetap anotasi jika nominal terbaca
- Tertutup Sebagian: anotasi jika minimal 75% terlihat
- Blur/Tidak Fokus: anotasi jika nominal terbaca jelas
- Multiple Overlap: anotasi terpisah selama terlihat >75%

## 6. Tips Efisiensi
1. Gunakan shortcut keyboard Roboflow
2. Mulai dari gambar pencahayaan normal
3. Validasi per batch (50-100 gambar)
4. Catat kasus unik untuk evaluasi

## 7. Preprocessing di Roboflow
- Auto-Orient: ON
- Resize: 640x640 (sesuai config YOLOv5)
- Maintain Aspect Ratio: YES
- Global Contrast Normalization: OFF (handled by model)