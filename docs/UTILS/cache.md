# Dokumentasi Modul Cache SmartCash

Dokumen ini berisi dokumentasi komprehensif untuk modul cache yang diimplementasikan sebagai bagian dari sistem SmartCash. Modul ini didesain untuk optimasi penyimpanan sementara hasil preprocessing dan komputasi dengan fitur-fitur canggih.

## Daftar Isi

1. [Pengenalan](#pengenalan)
2. [Struktur Modul](#struktur-modul)
3. [Cara Penggunaan](#cara-penggunaan)
4. [API Reference](#api-reference)
5. [Contoh Penggunaan](#contoh-penggunaan)
6. [Konfigurasi](#konfigurasi)
7. [Migrasi dari EnhancedCache](#migrasi-dari-enhancedcache)
8. [Troubleshooting](#troubleshooting)

## Pengenalan

Modul cache SmartCash dirancang untuk mengoptimalkan kinerja aplikasi dengan menyimpan hasil operasi yang membutuhkan waktu atau komputasi tinggi. Dengan menggunakan sistem cache ini, aplikasi dapat:

- Mengurangi waktu pemrosesan untuk operasi yang berulang
- Menghemat sumber daya komputasi
- Meningkatkan responsivitas aplikasi
- Memonitor penggunaan cache dan performa

Implementasi cache ini menyediakan fitur-fitur canggih termasuk:
- **Time-To-Live (TTL)**: Entri cache kedaluwarsa setelah periode tertentu
- **Garbage Collection**: Pembersihan otomatis untuk entri kedaluwarsa dan mengelola ukuran cache
- **Threading Safety**: Aman digunakan dalam lingkungan multi-threading
- **Monitoring Performa**: Statistik lengkap tentang hit, miss, dan penghematan

## Struktur Modul

Modul cache terstruktur menjadi beberapa komponen yang memiliki tanggung jawab spesifik:

| Komponen | Deskripsi |
|----------|-----------|
| `CacheManager` | Kelas utama untuk mengkoordinasikan semua operasi cache |
| `CacheIndex` | Mengelola metadata cache dan persistensi index |
| `CacheStorage` | Menangani operasi penyimpanan dan pembacaan data cache |
| `CacheCleanup` | Mengelola pembersihan cache dan verifikasi integritas |
| `CacheStats` | Mengumpulkan dan melaporkan statistik penggunaan cache |

Diagram hubungan komponen:

```
+-------------------+
|   CacheManager    |
+-------------------+
          |
          v
 +----------------+    +----------------+    +----------------+
 |   CacheIndex   |<-->|  CacheStorage  |<-->|  CacheCleanup  |
 +----------------+    +----------------+    +----------------+
          ^                    ^                    ^
          |                    |                    |
          +--------------------+--------------------+
                              |
                     +----------------+
                     |   CacheStats   |
                     +----------------+
```

## Cara Penggunaan

### Instalasi

Modul cache adalah bagian dari sistem SmartCash dan tidak memerlukan instalasi terpisah. Modul ini tersedia secara otomatis ketika SmartCash diinstal.

### Import Dasar

```python
# Menggunakan CacheManager (rekomendasi untuk kode baru)
from smartcash.utils.cache import CacheManager

# Jika perlu kompatibilitas dengan kode lama
from smartcash.utils.enhanced_cache import EnhancedCache
```

### Inisialisasi

```python
# Inisialisasi dengan parameter default
cache = CacheManager()

# Inisialisasi dengan parameter kustom
cache = CacheManager(
    cache_dir=".cache/my_project",
    max_size_gb=2.0,
    ttl_hours=48,
    auto_cleanup=True,
    cleanup_interval_mins=60,
    logger=custom_logger
)
```

### Operasi Dasar

```python
# Menyimpan data ke cache
key = cache.get_cache_key(file_path, params)
cache.put(key, data)

# Mengambil data dari cache
data = cache.get(key)

# Memeriksa keberadaan key
if cache.exists(key):
    # Key ada dan valid
    pass
```

### Pembersihan dan Pemeliharaan

```python
# Bersihkan cache berdasarkan TTL
cache.cleanup(expired_only=True)

# Bersihkan cache untuk mengurangi ukuran
cache.cleanup(force=True)

# Hapus seluruh isi cache
cache.clear()

# Verifikasi dan perbaiki integritas cache
cache.verify_integrity(fix=True)
```

### Statistik

```python
# Dapatkan statistik penggunaan cache
stats = cache.get_stats()
print(f"Hit ratio: {stats['hit_ratio']:.2f}%")
print(f"Total saved: {stats['saved_bytes']/1024/1024:.2f} MB")
```

## API Reference

### CacheManager

Kelas utama untuk mengelola operasi cache.

```python
class CacheManager:
    def __init__(
        self,
        cache_dir: str = ".cache/preprocessing",
        max_size_gb: float = 1.0,
        ttl_hours: int = 24,
        auto_cleanup: bool = True,
        cleanup_interval_mins: int = 30,
        logger: Optional[SmartCashLogger] = None
    )
```

| Metode | Deskripsi |
|--------|-----------|
| `get_cache_key(file_path, params)` | Menghasilkan key cache berdasarkan path file dan parameter |
| `get(key, measure_time=True)` | Ambil data dari cache menggunakan key |
| `exists(key)` | Periksa apakah key ada dan valid di cache |
| `put(key, data, estimated_size=None)` | Simpan data ke cache dengan key tertentu |
| `cleanup(expired_only=False, force=False)` | Bersihkan cache, opsional hanya yang kedaluwarsa |
| `clear()` | Hapus seluruh isi cache |
| `get_stats()` | Dapatkan statistik penggunaan cache |
| `verify_integrity(fix=True)` | Verifikasi dan opsional perbaiki integritas cache |

### CacheIndex

Mengelola metadata cache dan persistensi index.

```python
class CacheIndex:
    def __init__(
        self, 
        cache_dir: Path, 
        logger: Optional[SmartCashLogger] = None
    )
```

| Metode | Deskripsi |
|--------|-----------|
| `load_index()` | Muat index cache dari disk |
| `save_index()` | Simpan index cache ke disk |
| `get_files()` | Dapatkan semua file dalam index |
| `get_file_info(key)` | Dapatkan info file dari index |
| `add_file(key, size)` | Tambahkan file ke index |
| `remove_file(key)` | Hapus file dari index |
| `update_access_time(key)` | Perbarui waktu akses file |
| `update_cleanup_time()` | Perbarui waktu pembersihan terakhir |
| `get_total_size()` | Dapatkan total ukuran cache |
| `set_total_size(size)` | Set total ukuran cache |

### CacheStorage

Menangani operasi penyimpanan dan pembacaan data cache.

```python
class CacheStorage:
    def __init__(
        self, 
        cache_dir: Path, 
        logger: Optional[SmartCashLogger] = None
    )
```

| Metode | Deskripsi |
|--------|-----------|
| `create_cache_key(file_path, params)` | Buat key cache dari file path dan parameter |
| `save_to_cache(cache_path, data)` | Simpan data ke file cache |
| `load_from_cache(cache_path, measure_time=True)` | Muat data dari file cache |
| `delete_file(cache_path)` | Hapus file cache |

### CacheCleanup

Mengelola pembersihan cache dan verifikasi integritas.

```python
class CacheCleanup:
    def __init__(
        self, 
        cache_dir: Path,
        cache_index,
        max_size_bytes: int,
        ttl_seconds: int,
        cleanup_interval: int,
        cache_stats,
        logger: Optional[SmartCashLogger] = None
    )
```

| Metode | Deskripsi |
|--------|-----------|
| `setup_auto_cleanup()` | Setup thread untuk pembersihan otomatis |
| `cleanup(expired_only=False, force=False)` | Bersihkan cache berdasarkan kriteria |
| `clear_all()` | Bersihkan seluruh cache |
| `verify_integrity(fix=True)` | Verifikasi dan perbaiki integritas cache |

### CacheStats

Mengumpulkan dan melaporkan statistik penggunaan cache.

```python
class CacheStats:
    def __init__(
        self, 
        logger: Optional[SmartCashLogger] = None
    )
```

| Metode | Deskripsi |
|--------|-----------|
| `reset()` | Reset semua statistik ke nilai awal |
| `update_hits()` | Tambahkan cache hit |
| `update_misses()` | Tambahkan cache miss |
| `update_evictions()` | Tambahkan cache eviction |
| `update_expired()` | Tambahkan cache expired |
| `update_saved_bytes(bytes_saved)` | Perbarui jumlah byte yang disimpan |
| `update_saved_time(time_saved)` | Perbarui waktu yang dihemat |
| `get_raw_stats()` | Dapatkan statistik mentah |
| `get_all_stats(cache_dir, cache_index, max_size_bytes)` | Dapatkan semua statistik cache |

## Contoh Penggunaan

### Contoh 1: Caching Hasil Preprocessing Gambar

```python
import cv2
from smartcash.utils.cache import CacheManager

# Inisialisasi cache
cache = CacheManager(
    cache_dir=".cache/image_processing",
    max_size_gb=5.0,
    ttl_hours=72
)

def process_image(image_path, params):
    # Buat cache key
    cache_key = cache.get_cache_key(image_path, params)
    
    # Cek apakah hasil sudah ada di cache
    processed_image = cache.get(cache_key)
    
    if processed_image is not None:
        print("✅ Hasil ditemukan di cache!")
        return processed_image
    
    print("🔄 Memproses gambar...")
    
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Lakukan preprocessing (contoh)
    if params.get('resize'):
        image = cv2.resize(image, params['resize'])
    
    if params.get('blur'):
        image = cv2.GaussianBlur(image, (5, 5), params['blur'])
    
    # Proses lainnya...
    
    # Simpan hasil ke cache
    cache.put(cache_key, image)
    
    return image
```

### Contoh 2: Penggunaan dengan Custom Logger

```python
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.cache import CacheManager

# Buat logger
logger = SmartCashLogger(
    name="preprocessing_cache",
    log_to_file=True,
    log_dir="logs/cache"
)

# Inisialisasi cache dengan logger
cache = CacheManager(
    cache_dir=".cache/preprocessing",
    max_size_gb=2.0,
    logger=logger
)

# Selanjutnya gunakan cache seperti biasa
```

### Contoh 3: Monitoring Performa Cache

```python
from smartcash.utils.cache import CacheManager
import time

cache = CacheManager()

# Simulasi beberapa operasi
for i in range(100):
    key = f"test_key_{i}"
    cache.put(key, f"data_{i}")

# Simulasi beberapa hit dan miss
for i in range(150):
    key = f"test_key_{i % 120}"  # Akan menghasilkan beberapa miss
    data = cache.get(key)

# Dapatkan dan tampilkan statistik
stats = cache.get_stats()

print("📊 Statistik Performa Cache:")
print(f"   Hits: {stats['hits']}")
print(f"   Misses: {stats['misses']}")
print(f"   Hit Ratio: {stats['hit_ratio']:.2f}%")
print(f"   Cache Size: {stats['cache_size_mb']:.2f} MB")
print(f"   Usage: {stats['usage_percent']:.2f}% dari maksimum")
```

## Konfigurasi

Berikut penjelasan detail parameter konfigurasi:

### Parameter CacheManager

| Parameter | Tipe | Default | Deskripsi |
|-----------|------|---------|-----------|
| `cache_dir` | `str` | ".cache/preprocessing" | Direktori penyimpanan file cache |
| `max_size_gb` | `float` | 1.0 | Ukuran maksimum cache dalam GB |
| `ttl_hours` | `int` | 24 | Waktu hidup entri cache dalam jam |
| `auto_cleanup` | `bool` | True | Aktifkan pembersihan otomatis |
| `cleanup_interval_mins` | `int` | 30 | Interval pembersihan otomatis dalam menit |
| `logger` | `SmartCashLogger` | None | Logger kustom (optional) |

### Pengaturan Lanjutan

Beberapa pengaturan dapat diubah melalui properti kelas setelah inisialisasi:

```python
cache = CacheManager()

# Ubah nilai TTL
cache.ttl_seconds = 12 * 3600  # Ubah ke 12 jam

# Ubah interval pembersihan
cache.cleanup_interval = 15 * 60  # Ubah ke 15 menit
```

## Migrasi dari EnhancedCache

Jika Anda sebelumnya menggunakan `EnhancedCache`, berikut panduan migrasi ke `CacheManager`:

### Perubahan Import

```python
# Kode lama
from smartcash.utils.enhanced_cache import EnhancedCache

# Kode baru
from smartcash.utils.cache import CacheManager
```

### Inisialisasi

Nama parameter sama persis antara `EnhancedCache` dan `CacheManager`, jadi tidak ada perubahan diperlukan.

```python
# Kode lama
cache = EnhancedCache(cache_dir="path/to/cache", max_size_gb=2.0)

# Kode baru
cache = CacheManager(cache_dir="path/to/cache", max_size_gb=2.0)
```

### Kompatibilitas Mundur

Untuk menjaga kompatibilitas dengan kode yang sudah ada, `EnhancedCache` tetap tersedia sebagai alias untuk `CacheManager`. Dengan demikian, kode yang sudah ada akan tetap berfungsi tanpa perubahan:

```python
# Kode ini tetap berfungsi
from smartcash.utils.enhanced_cache import EnhancedCache
cache = EnhancedCache()
```

## Troubleshooting

### File Cache Rusak

Jika file cache rusak atau index tidak konsisten:

```python
# Verifikasi dan perbaiki cache
cache.verify_integrity(fix=True)
```

### Error Thread-Safety

Jika mengalami masalah terkait multi-threading:
- Pastikan versi modul cache terbaru digunakan
- Hindari memodifikasi objek cache secara langsung antar thread
- Gunakan instance cache terpisah untuk thread terpisah jika diperlukan

### Penggunaan Memori Tinggi

Jika cache menggunakan terlalu banyak memori:

```python
# Kurangi ukuran maksimum cache
cache = CacheManager(max_size_gb=0.5)

# Paksa pembersihan untuk mengurangi ukuran
cache.cleanup(force=True)
```

### Log Debug yang Lebih Detail

Untuk mendapatkan informasi debug yang lebih detail:

```python
import logging
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.cache import CacheManager

debug_logger = SmartCashLogger(
    "cache_debug",
    level=logging.DEBUG
)

cache = CacheManager(logger=debug_logger)
```


