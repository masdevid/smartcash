
# Evaluasi Model Deteksi Nilai Mata Uang

Evaluasi dilakukan menggunakan beberapa metrik untuk memastikan model dapat mendeteksi nilai mata uang dengan baik. Skenario pengujian dirancang untuk mengevaluasi kinerja model YOLOv5 yang dioptimalkan dengan EfficientNet-B4 dalam mendeteksi nilai mata uang dan membandingkan performa model pada berbagai skenario untuk mengukur Akurasi, Precision, Recall, F1-Score, mAP dan kecepatan inferensi model.

## Rancangan Skema Pengujian

Berikut adalah rancangan skema pengujian untuk menguji Optimalisasi YOLOv5 dengan Arsitektur EfficientNet-B4 dalam system deteksi nilai mata uang:

1. **Variable yang akan diuji**
   - a. YOLOv5 dengan CSPDarknet sebagai backbone default YOLOV5
   - b. Integrasi YOLOv5 dengan arsitektur EfficientNet-B4 sebagai backbone

2. **Dataset**
   Menggunakan Dataset uang kertas emisi 2022, dengan berbagai posisi pengambilan gambar dan pencahayaan.

3. **Parameter evaluasi:**
   - a. Akurasi
   - b. Precision
   - c. Recall
   - d. F1-Score
   - e. mAP
   - f. Waktu inferensi (inference time)

4. **Skema scenario pengujian**
   Pengujian dilakukan dengan 4 skenario utama, masing-masing dengan parameter atau kondisi tertentu:
   - a. **Skenario-1:** Deteksi nilai uang dengan model YOLOv5 Default (CSPDarknet) sebagai baseline backbone dengan input gambar uang kertas rupiah dengan posisi pengambilan gambar yang bervariasi, misalnya dari atas, bawah, samping kiri dan kanan.
   - b. **Skenario-2:** Deteksi nilai uang dengan model YOLOv5 Default (CSPDarknet) sebagai baseline backbone dengan input gambar uang kertas rupiah dengan pencahayaan beravariasi.
   - c. **Skenario-3:** Deteksi nilai uang dengan model YOLOv5 dengan arsitektur EfficientNet-B4 sebagai backbone dengan input gambar uang kertas rupiah dengan posisi pengambilan gambar yang bervariasi, misalnya dari atas, bawah, samping kiri dan kanan.
   - d. **Skenario-4:** Deteksi nilai uang dengan model YOLOv5 dengan arsitektur EfficientNet-B4 sebagai backbone dengan input gambar uang kertas rupiah dengan pencahayaan beravariasi.

5. **Pengaturan eksperimen**
   - a. Setiap skenario diuji sebanyak 3 kali eksperimen, untuk memastikan hasil yang konsisten dan valid.
   - b. Dataset dibagi menjadi:
     - **Training set (70%):** Data ini digunakan untuk melatih model agar model dapat mempelajari pola dan fitur dari setiap gambar uang kertas rupiah.
     - **Validation set (15%):** Data ini digunakan untuk mengatur dan mengevaluasi model selama pelatihan.
     - **Testing set (15%):** Data ini digunakan untuk menguji model. Model yang telah dilatih akan diuji pada data ini untuk mengevaluasi seberapa baik model tersebut dapat mendeteksi data baru yang belum pernah dilihat sebelumnya.

6. **Proses pengujian**
   - a. Melatih YOLOv5 dengan backbone sesuai skenario pengujian.
   - b. Menghitung metrik: Akurasi, Precision, Recall, F1-Score dan mAP
   - c. Menggunakan data testing untuk mengukur kinerja model pada data baru.
   - d. Mengukur waktu yang dibutuhkan model untuk memproses satu gambar (latency).

7. **Output eksperimen**
   Hasil setiap skenario pengujian dicatat dalam tabel performa sebagai berikut:

   | Skenario       | Akurasi | Precision | Recall | F1-Score | mAP | Waktu Inferensi |
   |----------------|---------|-----------|--------|----------|-----|-----------------|
   | Skenario-1     |         |           |        |          |     |                 |
   | Skenario-2     |         |           |        |          |     |                 |
   | Skenario-3     |         |           |        |          |     |                 |
   | Skenario-4     |         |           |        |          |     |                 |

Melalui tabel performa ini dapat diketahui bagaimana optimalisasi YOLOv5 dengan arsitektur EfficientNet-B4 dalam sistem deteksi nilai mata uang kertas rupiah.
