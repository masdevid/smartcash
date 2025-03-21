# SmartCash v2 - Domain Components

## 1. File Structure
```
smartcash/components/
├── init.py                 # Ekspor komponen reusable
├── observer/                   # Observer pattern
│   ├── init.py             # Ekspor komponen observer pattern
│   ├── base_observer.py        # BaseObserver: Kelas dasar untuk semua observer
│   ├── compatibility_observer.py # CompatibilityObserver: Adapter untuk observer lama
│   ├── decorators_observer.py  # Dekorator @observable dan @observe
│   ├── event_dispatcher_observer.py # EventDispatcher: Dispatcher untuk notifikasi
│   ├── event_registry_observer.py # EventRegistry: Registry untuk observer
│   ├── event_topics_observer.py # EventTopics: Topik event standar
│   ├── manager_observer.py     # ObserverManager: Manager untuk observer
│   ├── priority_observer.py    # ObserverPriority: Prioritas observer
│   └── cleanup_observer.py     # CleanupObserver: Pembersihan observer saat exit
└── cache/                      # Cache pattern
├── init.py             # Ekspor komponen cache
├── cleanup_cache.py        # CacheCleanup: Pembersihan cache otomatis
├── indexing_cache.py       # CacheIndex: Pengindeksan cache
├── manager_cache.py        # CacheManager: Manager cache utama
├── stats_cache.py          # CacheStats: Statistik cache
└── storage_cache.py        # CacheStorage: Penyimpanan cache

## 2. Class and Methods Mapping

### EventDispatcher (event_dispatcher_observer.py)
- **Fungsi**: Mengelola notifikasi event ke observer dengan thread-safety dan dukungan asinkron
- **Metode Utama**:
  - `register(event_type, observer, priority)`: Daftarkan observer untuk tipe event
  - `register_many(event_types, observer, priority)`: Daftarkan observer untuk beberapa event
  - `notify(event_type, sender, async_mode, **kwargs)`: Kirim notifikasi ke semua observer
  - `unregister(event_type, observer)`: Batalkan registrasi observer
  - `unregister_from_all(observer)`: Batalkan registrasi dari semua event
  - `unregister_all(event_type)`: Hapus semua observer dari registry
  - `wait_for_async_notifications(timeout)`: Tunggu notifikasi asinkron selesai
  - `get_stats()`: Dapatkan statistik dispatcher
  - `enable_logging()`, `disable_logging()`: Aktifkan/nonaktifkan logging
  - `shutdown()`: Bersihkan resources dispatcher

### EventRegistry (event_registry_observer.py)
- **Fungsi**: Registry thread-safe dengan weak reference untuk mencegah memory leak
- **Metode Utama**:
  - `register(event_type, observer, priority)`: Daftarkan observer untuk event
  - `unregister(event_type, observer)`: Hapus observer dari registry
  - `unregister_from_all(observer)`: Hapus observer dari semua event
  - `unregister_all(event_type)`: Hapus semua observer untuk event atau semua event
  - `get_observers(event_type)`: Dapatkan observer untuk event, termasuk hierarki event
  - `get_all_event_types()`: Dapatkan semua tipe event terdaftar
  - `get_stats()`: Dapatkan statistik registry
  - `clean_references()`: Bersihkan weak references yang tidak valid

### BaseObserver (base_observer.py)
- **Fungsi**: Kelas dasar untuk semua observer dengan filter event dan prioritas
- **Metode Utama**:
  - `update(event_type, sender, **kwargs)`: Handler event (abstract)
  - `should_process_event(event_type)`: Cek filter event (string/regex/list/callback)
  - `enable()`, `disable()`: Aktifkan/nonaktifkan observer
  - `__lt__(other)`: Support pengurutan berdasarkan prioritas
  - `__eq__(other)`, `__hash__()`: Dukungan untuk set/dict dengan ID unik

### ObserverManager (manager_observer.py)
- **Fungsi**: Mengelola observer dengan sistem grup untuk lifecycle management
- **Metode Utama**:
  - `create_simple_observer(event_type, callback, name, priority, group)`: Buat observer sederhana
  - `create_observer(observer_class, event_type, name, priority, group, **kwargs)`: Buat observer custom
  - `create_progress_observer(event_types, total, desc, use_tqdm, callback, group)`: Buat observer progress
  - `create_logging_observer(event_types, log_level, format_string, group)`: Buat observer untuk logging
  - `get_observers_by_group(group)`: Dapatkan observer dalam grup
  - `unregister_group(group)`: Batalkan registrasi grup observer
  - `unregister_all()`: Batalkan registrasi semua observer
  - `get_stats()`: Dapatkan statistik manager

### CompatibilityObserver (compatibility_observer.py)
- **Fungsi**: Adapter untuk observer lama (non-BaseObserver)
- **Metode Utama**:
  - `update(event_type, sender, **kwargs)`: Mapping ke metode observer lama
  - `_map_event_to_method(event_type)`: Konversi tipe event ke nama metode
  - `_check_on_event_methods()`: Periksa metode on_* yang tersedia

### Decorators (decorators_observer.py)
- **Fungsi**: Dekorator untuk observer pattern
- **Dekorator Utama**:
  - `@observable(event_type, include_args, include_result)`: Tandai metode sebagai observable
  - `@observe(event_types, priority, event_filter)`: Buat kelas menjadi observer

### EventTopics (event_topics_observer.py)
- **Fungsi**: Definisi konstanta untuk topik event standar
- **Konstanta Utama**:
  - `Kategori`: TRAINING, EVALUATION, DETECTION, dll.
  - `Events terstruktur`: TRAINING_START, EPOCH_END, BATCH_START, dll.
  - `get_all_topics()`: Dapatkan semua topik event yang didefinisikan

### ObserverPriority (priority_observer.py)
- **Fungsi**: Konstanta untuk prioritas observer
- **Konstanta**:
  - `CRITICAL=100, HIGH=75, NORMAL=50, LOW=25, LOWEST=0`

### CleanupObserver (cleanup_observer.py)
- **Fungsi**: Pembersihan otomatis observer saat aplikasi selesai
- **Metode Utama**:
  - `register_observer_manager(manager)`: Daftarkan manager untuk dibersihkan
  - `register_cleanup_function(cleanup_func)`: Daftarkan fungsi cleanup
  - `cleanup_observer_group(manager, group)`: Bersihkan grup observer
  - `cleanup_all_observers()`: Bersihkan semua observer (atexit handler)
  - `register_notebook_cleanup(observer_managers, cleanup_functions)`: Setup cleanup untuk notebook

### CacheManager (manager_cache.py)
- **Fungsi**: Koordinator sistem caching terpadu dengan cleanup otomatis
- **Metode Utama**:
  - `get(key, measure_time)`: Ambil data dari cache dengan statistik
  - `put(key, data, estimated_size)`: Simpan data ke cache
  - `get_cache_key(file_path, params)`: Buat kunci cache dari file dan parameter
  - `cleanup(expired_only, force)`: Bersihkan cache
  - `clear()`: Hapus seluruh cache
  - `get_stats()`: Dapatkan statistik lengkap cache
  - `verify_integrity(fix)`: Periksa dan perbaiki integritas cache

### CacheStorage (storage_cache.py)
- **Fungsi**: Menyimpan dan memuat data cache dari disk dengan pengukuran performa
- **Metode Utama**:
  - `create_cache_key(file_path, params)`: Buat key unik berbasis hash
  - `save_to_cache(cache_path, data)`: Simpan data ke file
  - `load_from_cache(cache_path, measure_time)`: Muat data dengan timing
  - `delete_file(cache_path)`: Hapus file cache

### CacheIndex (indexing_cache.py)
- **Fungsi**: Mengelola metadata cache dengan atomic update
- **Metode Utama**:
  - `load_index()`: Muat index dari disk
  - `save_index()`: Simpan index ke disk secara atomic
  - `get_files()`: Dapatkan semua file terdaftar
  - `get_file_info(key)`: Dapatkan info file cache
  - `add_file(key, size)`: Tambahkan file ke index
  - `remove_file(key)`: Hapus file dari index
  - `update_access_time(key)`: Update waktu akses terakhir
  - `get/set_total_size()`: Get/set ukuran total cache

### CacheCleanup (cleanup_cache.py)
- **Fungsi**: Pembersihan cache otomatis dengan thread worker
- **Metode Utama**:
  - `setup_auto_cleanup()`: Jalankan thread worker otomatis
  - `cleanup(expired_only, force)`: Bersihkan cache dengan strategi expired/LRU
  - `_identify_expired_files(current_time)`: Identifikasi file kadaluarsa
  - `_remove_by_lru(cleanup_stats, already_removed)`: Hapus file dengan LRU
  - `clear_all()`: Hapus seluruh cache
  - `verify_integrity(fix)`: Validasi dan perbaiki integritas cache

### CacheStats (stats_cache.py)
- **Fungsi**: Melacak dan menghitung statistik penggunaan cache
- **Metode Utama**:
  - `reset()`: Reset semua statistik
  - `update_hits()`: Tambah hit count
  - `update_misses()`: Tambah miss count
  - `update_evictions()`: Tambah eviction count
  - `update_expired()`: Tambah expired count
  - `update_saved_bytes(bytes_saved)`: Tambah byte tersimpan
  - `update_saved_time(time_saved)`: Tambah waktu tersimpan
  - `get_raw_stats()`: Dapatkan raw stats
  - `get_all_stats(cache_dir, cache_index, max_size_bytes)`: Hitung dan validasi semua statistik
  ```