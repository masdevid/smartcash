# Worker Utilities Module

## Overview
Modul `worker_utils` menyediakan utilitas untuk mengoptimalkan jumlah worker thread berdasarkan jenis operasi dan spesifikasi sistem. Modul ini membantu memastikan penggunaan thread yang efisien di seluruh kodebase SmartCash.

## Daftar Isi
- [Fungsi Utama](#fungsi-utama)
  - [get_optimal_worker_count](#get_optimal_worker_count)
  - [get_worker_counts_for_operations](#get_worker_counts_for_operations)
  - [get_file_operation_workers](#get_file_operation_workers)
  - [get_download_workers](#get_download_workers)
  - [get_rename_workers](#get_rename_workers)
  - [safe_worker_count](#safe_worker_count)
- [Fungsi Bantuan](#fungsi-bantuan)
  - [optimal_io_workers](#optimal_io_workers)
  - [optimal_cpu_workers](#optimal_cpu_workers)
  - [optimal_mixed_workers](#optimal_mixed_workers)
- [Contoh Penggunaan](#contoh-penggunaan)
- [Best Practices](#best-practices)

## Fungsi Utama

### get_optimal_worker_count
```python
get_optimal_worker_count(operation_type: OperationType = 'io') -> int
```

Menghitung jumlah worker optimal berdasarkan jenis operasi.

#### Parameter
- `operation_type`: Jenis operasi (default: 'io')
  - `'io'` atau `'io_bound'`: Operasi I/O bound (jaringan, disk)
  - `'cpu'` atau `'cpu_bound'`: Operasi CPU bound (pemrosesan)
  - `'mixed'`: Campuran antara I/O dan CPU

#### Contoh
```python
# Untuk operasi I/O bound
workers = get_optimal_worker_count('io')

# Untuk operasi CPU bound
workers = get_optimal_worker_count('cpu')
```

### get_worker_counts_for_operations
```python
get_worker_counts_for_operations() -> Dict[str, int]
```

Mendapatkan jumlah worker optimal untuk semua operasi standar.

#### Contoh
```python
counts = get_worker_counts_for_operations()
# {'download': 5, 'validation': 4, 'uuid_renaming': 4, ...}
```

### get_file_operation_workers
```python
get_file_operation_workers(file_count: int) -> int
```

Menghitung worker optimal untuk operasi file berdasarkan jumlah file.

#### Parameter
- `file_count`: Jumlah file yang akan diproses

#### Contoh
```python
workers = get_file_operation_workers(1000)  # Mengembalikan worker optimal untuk 1000 file
```

### get_download_workers
```python
get_download_workers() -> int
```

Mendapatkan jumlah worker optimal untuk operasi download.

#### Contoh
```python
workers = get_download_workers()
```

### get_rename_workers
```python
get_rename_workers(total_files: int) -> int
```

Menghitung worker optimal untuk operasi pengubahan nama file.

#### Parameter
- `total_files`: Total file yang akan diubah namanya

#### Contoh
```python
workers = get_rename_workers(1000)
```

### safe_worker_count
```python
safe_worker_count(count: int) -> int
```

Memastikan jumlah worker dalam batas aman (1-8).

#### Parameter
- `count`: Jumlah worker yang diminta

#### Contoh
```python
safe_count = safe_worker_count(10)  # Akan mengembalikan 8
```

## Fungsi Bantuan

### optimal_io_workers
```python
optimal_io_workers() -> int
```

Alias untuk `get_optimal_worker_count('io')`.

### optimal_cpu_workers
```python
optimal_cpu_workers() -> int
```

Alias untuk `get_optimal_worker_count('cpu')`.

### optimal_mixed_workers
```python
optimal_mixed_workers() -> int
```

Alias untuk `get_optimal_worker_count('mixed')`.

## Contoh Penggunaan

### Contoh 1: Menggunakan Worker untuk Operasi File
```python
from smartcash.common.worker_utils import get_file_operation_workers
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    # Proses file di sini
    pass

files = ["file1.txt", "file2.txt", "file3.txt"]
workers = get_file_operation_workers(len(files))

with ThreadPoolExecutor(max_workers=workers) as executor:
    executor.map(process_file, files)
```

### Contoh 2: Menggunakan Worker untuk Download
```python
from smartcash.common.worker_utils import get_download_workers
import requests
from concurrent.futures import ThreadPoolExecutor

def download_file(url):
    response = requests.get(url)
    # Simpan file
    pass

urls = ["http://example.com/file1", "http://example.com/file2"]
workers = get_download_workers()

with ThreadPoolExecutor(max_workers=workers) as executor:
    executor.map(download_file, urls)
```

## Best Practices

1. **Pilih Jenis Worker yang Tepat**
   - Gunakan `'io'` untuk operasi I/O bound (file, jaringan)
   - Gunakan `'cpu'` untuk operasi CPU bound (pemrosesan)
   - Gunakan `'mixed'` untuk operasi campuran

2. **Gunakan Fungsi Spesifik**
   - Gunakan `get_download_workers()` untuk download
   - Gunakan `get_file_operation_workers()` untuk operasi file
   - Gunakan `get_rename_workers()` untuk mengubah nama file

3. **Batasi Jumlah Worker**
   - Gunakan `safe_worker_count()` untuk memastikan jumlah worker dalam batas aman
   - Hindari membuat terlalu banyak worker untuk menghindari overhead

4. **Gunakan ThreadPoolExecutor**
   - Selalu gunakan `ThreadPoolExecutor` dengan worker yang dioptimalkan
   - Tutup executor dengan benar menggunakan `with` statement

5. **Monitor Kinerja**
   - Pantau penggunaan CPU dan memori saat menggunakan worker
   - Sesuaikan jumlah worker jika diperlukan berdasarkan pengamatan kinerja

Dokumentasi ini mencakup semua fitur utama dari modul worker_utils. Untuk informasi lebih lanjut, lihat kode sumber di `smartcash/common/worker_utils.py`.
