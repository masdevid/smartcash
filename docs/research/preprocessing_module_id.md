# Preprocessing Module SmartCash

## Deskripsi

Modul Preprocessing SmartCash menyediakan solusi komprehensif untuk memproses, membersihkan, dan menyiapkan data sebelum digunakan dalam pelatihan atau evaluasi model machine learning. Dirancang khusus untuk lingkungan Jupyter/Colab, modul ini memudahkan pengguna dalam melakukan transformasi data secara efisien dan aman.

Setiap proses preprocessing didukung oleh antarmuka pengguna yang intuitif, memungkinkan konfigurasi parameter seperti augmentasi, normalisasi, dan pembagian dataset (train/valid/test) secara fleksibel. Modul ini mendukung berbagai format data populer dan dapat melakukan konversi otomatis sesuai kebutuhan framework yang digunakan.

Keamanan dan integritas data menjadi prioritas utama. Setiap operasi yang berpotensi mengubah atau menghapus data akan meminta konfirmasi eksplisit dari pengguna. Sistem log terintegrasi mencatat setiap langkah preprocessing, memberikan transparansi penuh dan kemudahan audit.

Modul ini mendukung berbagai teknik preprocessing seperti resize, crop, rotasi, flipping, normalisasi, dan augmentasi berbasis konfigurasi. Proses dapat dijalankan secara batch dengan optimasi multi-thread untuk mempercepat eksekusi pada dataset berukuran besar. Validasi struktur dan konsistensi data dilakukan secara otomatis setelah setiap operasi.

Manajemen versi preprocessing memungkinkan pengguna melacak perubahan dan kembali ke versi sebelumnya jika diperlukan. Metadata lengkap, termasuk parameter dan waktu eksekusi, disimpan untuk setiap proses. Modul juga menyediakan preview hasil preprocessing sebelum data disimpan secara permanen.

## Alur Kerja

```mermaid
sequenceDiagram
    participant P as Pengguna
    participant UI as Antarmuka
    participant PM as Preprocessing Manager
    participant PS as Preprocessing Service

    P->>UI: Konfigurasi Parameter Preprocessing
    activate UI

    UI->>PM: Validasi Konfigurasi
    activate PM

    PM->>PS: Persiapkan Proses
    activate PS
    PS-->>PM: Konfirmasi Siap
    deactivate PS

    PM->>PS: Mulai Preprocessing
    activate PS
    PS-->>PM: Update Progress
    PM-->>UI: Tampilkan Progress
    PS-->>PM: Selesai
    deactivate PS

    PM-->>UI: Tampilkan Hasil & Preview
    deactivate PM

    UI-->>P: Notifikasi Selesai
    deactivate UI
```

## Alur Operasi

Proses preprocessing dimulai dengan inisialisasi modul dan verifikasi parameter yang dimasukkan pengguna. Sistem akan menampilkan opsi konfigurasi seperti jenis augmentasi, metode normalisasi, dan pembagian dataset. Sebelum eksekusi, modul memeriksa integritas data dan meminta konfirmasi jika operasi akan mengubah data asli.

Setelah konfirmasi, proses preprocessing berjalan di latar belakang dengan optimasi multi-thread. Pengguna dapat memantau kemajuan melalui progress bar dan preview hasil secara real-time. Setiap langkah preprocessing divalidasi untuk memastikan hasil sesuai dengan konfigurasi.

Hasil preprocessing disimpan dalam struktur folder yang terorganisir, lengkap dengan metadata dan log operasi. Pengguna dapat mengakses riwayat proses, membandingkan hasil, dan mengembalikan data ke versi sebelumnya jika diperlukan. Modul juga mendukung export hasil preprocessing ke format yang kompatibel dengan berbagai framework machine learning.

## Diagram Urutan Operasi Preprocessing

```mermaid
sequenceDiagram
    participant User
    participant UI as Antarmuka Pengguna
    participant Handler
    participant Service
    participant Backend

    User->>UI: Masukkan Konfigurasi Preprocessing
    UI->>Handler: Validasi Input
    Handler->>Service: Persiapkan Proses
    Service->>Backend: Verifikasi Data
    Backend-->>Service: Konfirmasi
    Service-->>Handler: Siap Proses
    Handler-->>UI: Tampilkan Status

    loop Selama Proses
        Service->>Backend: Proses Data
        Backend-->>Service: Chunk Data
        Service-->>UI: Update Progress & Preview
    end

    Service->>Handler: Selesai
    Handler->>UI: Tampilkan Hasil
    UI-->>User: Notifikasi Selesai
```
