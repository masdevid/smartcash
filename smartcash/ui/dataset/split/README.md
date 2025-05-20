# SmartCash - Dataset Split UI

Modul ini bertujuan untuk menyediakan antarmuka konfigurasi split dataset untuk proyeksi SmartCash.

## Fitur

### UI Konfigurasi Split

UI ini memungkinkan pengguna untuk mengonfigurasi parameter-parameter berikut:
- Enabled/disabled split
- Train/validation/test ratio
- Random seed
- Stratify option

### Sinkronisasi Konfigurasi

Implementasi memastikan bahwa:
1. Konfigurasi disimpan dengan benar setelah tombol save ditekan
2. UI selalu konsisten dengan konfigurasi yang tersimpan
3. Reset mengembalikan konfigurasi dan UI ke nilai default
4. Perubahan status sinkronisasi ditampilkan di UI log

## Komponen Utama

- `split_initializer.py`: Inisialisasi UI dan menghubungkan komponen-komponen
- `components/split_components.py`: Komponen UI untuk split dataset
- `handlers/button_handlers.py`: Handler untuk button-button di UI
- `handlers/config_handlers.py`: Handler untuk operasi konfigurasi
- `handlers/sync_logger.py`: Utility untuk logging proses sinkronisasi di UI

## Testing

Untuk menjalankan test, gunakan perintah berikut:

```bash
python -m smartcash.ui.dataset.split.tests.run_standalone_tests
```

Test ini memverifikasi bahwa:
1. Konfigurasi disimpan dengan benar setelah save
2. UI diperbarui dengan benar setelah reset
3. UI dimuat dengan benar sesuai konfigurasi yang tersimpan

## Logging Sinkronisasi

Proses sinkronisasi dicatat dalam log UI dengan menggunakan `sync_logger.py`. Log ini menampilkan:
- Kapan sinkronisasi dimulai
- Status keberhasilan atau kegagalan sinkronisasi
- Detail error jika terjadi masalah

Contoh log:
- ✅ Konfigurasi berhasil disimpan
- ℹ️ Memulai sinkronisasi UI dari konfigurasi...
- ❌ Error saat menyimpan konfigurasi: [detail error] 