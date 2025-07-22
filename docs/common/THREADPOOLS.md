# Thread Pools Module

## Overview
Modul `threadpools` menyediakan utilitas untuk menjalankan tugas secara paralel menggunakan thread pools yang dioptimalkan. Modul ini dibangun di atas `concurrent.futures.ThreadPoolExecutor` dengan tambahan fitur seperti progress tracking dan pengumpulan statistik.

## Daftar Isi
- [Fungsi Utama](#fungsi-utama)
  - [process_in_parallel](#process_in_parallel)
  - [process_with_stats](#process_with_stats)
- [Fungsi Bantuan](#fungsi-bantuan)
  - [optimal_io_workers](#optimal_io_workers)
  - [optimal_cpu_workers](#optimal_cpu_workers)
  - [safe_worker_count](#safe_worker_count)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Best Practices](#best-practices)

## Fungsi Utama

### process_in_parallel
```python
process_in_parallel(
    items: List[Any],
    process_func: Callable[[Any], Any],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True
) -> List[Any]
```

Memproses daftar item secara paralel menggunakan thread pool.

#### Parameter
- `items`: Daftar item yang akan diproses
- `process_func`: Fungsi yang akan dipanggil untuk setiap item
- `max_workers`: Jumlah worker maksimum (default: optimal untuk I/O)
- `desc`: Deskripsi untuk progress bar
- `show_progress`: Tampilkan progress bar (default: True)

#### Contoh
```python
from smartcash.common.threadpools import process_in_parallel

def process_item(item):
    # Proses item di sini
    return f"Processed: {item}"

results = process_in_parallel(
    items=[1, 2, 3, 4, 5],
    process_func=process_item,
    desc="Processing items"
)
```

### process_with_stats
```python
process_with_stats(
    items: List[Any],
    process_func: Callable[[Any], Dict[str, int]],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True
) -> Dict[str, int]
```

Memproses item secara paralel dan mengumpulkan statistik.

#### Parameter
- `items`: Daftar item yang akan diproses
- `process_func`: Fungsi yang mengembalikan dictionary statistik
- `max_workers`: Jumlah worker maksimum (default: optimal untuk I/O)
- `desc`: Deskripsi untuk progress bar
- `show_progress`: Tampilkan progress bar (default: True)

#### Contoh
```python
from smartcash.common.threadpools import process_with_stats

def process_with_stat(item):
    # Proses item dan kembalikan statistik
    return {"processed": 1, "errors": 0 if item % 2 == 0 else 1}

stats = process_with_stats(
    items=[1, 2, 3, 4, 5],
    process_func=process_with_stat,
    desc="Processing with stats"
)
# Hasil: {'processed': 5, 'errors': 3}
```

## Fungsi Bantuan

### optimal_io_workers
```python
optimal_io_workers() -> int
```
Mengembalikan jumlah worker optimal untuk operasi I/O bound.

### optimal_cpu_workers
```python
optimal_cpu_workers() -> int
```
Mengembalikan jumlah worker optimal untuk operasi CPU bound.

### safe_worker_count
```python
safe_worker_count(count: int) -> int
```
Memastikan jumlah worker dalam batas aman (1-8).

## Contoh Penggunaan

### Contoh 1: Memproses File dengan Progress
```python
from pathlib import Path
from smartcash.common.threadpools import process_in_parallel

def process_image(file_path):
    # Proses gambar di sini
    return f"Processed {file_path.name}"

# Dapatkan daftar file gambar
image_files = list(Path("images").glob("*.jpg"))

# Proses secara paralel dengan progress bar
results = process_in_parallel(
    items=image_files,
    process_func=process_image,
    desc="Processing images"
)
```

### Contoh 2: Mengumpulkan Statistik Pemrosesan
```python
from smartcash.common.threadpools import process_with_stats

def process_document(doc_id):
    # Simulasikan pemrosesan dokumen
    if doc_id % 10 == 0:
        return {"processed": 1, "errors": 1, "empty": 0}
    return {"processed": 1, "errors": 0, "empty": 0}

# Proses 100 dokumen secara paralel
stats = process_with_stats(
    items=range(100),
    process_func=process_document,
    desc="Processing documents"
)

print(f"Total diproses: {stats.get('processed', 0)}")
print(f"Total error: {stats.get('errors', 0)}")
```

## Best Practices

1. **Pilih Jumlah Worker yang Tepat**
   - Gunakan `optimal_io_workers()` untuk operasi I/O bound
   - Gunakan `optimal_cpu_workers()` untuk operasi CPU bound
   - Gunakan `safe_worker_count()` untuk memastikan jumlah worker aman

2. **Gunakan Progress Bar**
   - Selalu sediakan deskripsi yang jelas untuk progress bar
   - Nonaktifkan progress bar (`show_progress=False`) untuk operasi yang sangat cepat

3. **Kelola Resource**
   - Hindari membuat terlalu banyak thread untuk menghindari overhead
   - Gunakan `max_workers` untuk membatasi jumlah thread jika diperlukan

4. **Error Handling**
   - Tangkap dan tangani exception di dalam fungsi yang diproses
   - Kembalikan statistik error untuk pemantauan

5. **Statistik yang Berguna**
   - Kumpulkan metrik yang relevan untuk memantau kinerja
   - Gunakan dictionary konsisten untuk statistik

Dokumentasi ini mencakup semua fitur utama dari modul threadpools. Untuk informasi lebih lanjut, lihat kode sumber di `smartcash/common/threadpools.py`.
