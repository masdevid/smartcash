# SmartCash v2 - Pemetaan Kelas dan Fungsi yang Diperbarui

## DOMAIN COMMON

### ConfigManager (smartcash/common/config.py)
- **Fungsi**: Mengelola konfigurasi aplikasi dengan dukungan dependency injection
- **Metode Utama**:
  - `load_config(config_file)`: Memuat konfigurasi dari file YAML/JSON
  - `get(key, default)`: Mengambil nilai dengan dot notation
  - `set(key, value)`: Mengatur nilai dengan dot notation
  - `merge_config(config)`: Menggabungkan konfigurasi dari dict/file
  - `save_config(config_file)`: Menyimpan konfigurasi ke file
  - `register(interface_type, implementation)`: Daftarkan implementasi
  - `register_instance(interface_type, instance)`: Daftarkan singleton
  - `register_factory(interface_type, factory)`: Daftarkan factory
  - `resolve(interface_type, *args, **kwargs)`: Resolve dependency
  - `sync_with_drive(config_file, sync_strategy)`: Sinkronisasi dengan Google Drive
  - `sync_all_configs(sync_strategy)`: Sinkronisasi semua file konfigurasi
  - `_merge_configs_smart(config1, config2)`: Gabungkan konfigurasi dengan strategi smart
- **Fungsi Global**:
  - `get_config_manager()`: Dapatkan singleton instance

### SmartCashLogger (smartcash/common/logger.py)
- **Fungsi**: Sistem logging dengan emoji dan warna
- **Metode Utama**:
  - `log(level, message)`: Mencatat pesan log
  - `add_callback(callback)`: Tambah callback untuk event log
  - `remove_callback(callback)`: Hapus callback dari logger
  - `debug/info/success/warning/error/critical()`: Shortcut log level
  - `progress(iterable)`: Membuat progress bar dengan tqdm
- **Class Tambahan**:
  - `LogLevel`: Enum untuk level log (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
- **Fungsi Global**:
  - `get_logger(name, level, log_file, use_colors, use_emojis, log_dir)`: Mendapatkan instance logger

### LayerConfigManager (smartcash/common/layer_config.py)
- **Fungsi**: Mengelola konfigurasi layer deteksi berdasarkan interface
- **Metode Utama**:
  - `get_layer_config(layer_name)`: Dapatkan konfigurasi spesifik layer
  - `get_layer_names()`: Dapatkan daftar semua nama layer 
  - `get_enabled_layers()`: Dapatkan layer yang aktif
  - `get_class_map()`: Mendapatkan mapping ID kelas ke nama kelas
  - `get_all_class_ids()`: Mendapatkan semua ID kelas dari semua layer
  - `get_layer_for_class_id(class_id)`: Mendapatkan nama layer untuk ID kelas
  - `load_config(config_path)`: Memuat konfigurasi dari file
  - `save_config(config_path)`: Menyimpan konfigurasi ke file
  - `update_layer_config(layer_name, config)`: Memperbarui konfigurasi layer
  - `set_layer_enabled(layer_name, enabled)`: Mengaktifkan/menonaktifkan layer
  - `validate_class_ids()`: Validasi ID kelas untuk mencegah duplikat
- **Pattern**: Implementasi singleton
- **Default Config**: Konfigurasi default untuk banknote, nominal, security
- **Fungsi Global**:
  - `get_layer_config()`: Fungsi helper untuk mendapatkan instance singleton

### EnvironmentManager (smartcash/common/environment.py)
- **Fungsi**: Mengelola deteksi dan konfigurasi lingkungan aplikasi
- **Metode Utama**:
  - `mount_drive(mount_path)`: Mount Google Drive jika di Colab
  - `get_path(relative_path)`: Mendapatkan path absolut
  - `get_project_root()`: Mendapatkan direktori root proyek
  - `setup_project_structure(use_drive)`: Membuat struktur direktori proyek
  - `create_symlinks()`: Membuat symlink dari direktori lokal ke Drive
  - `get_directory_tree(root_dir, max_depth)`: Mendapatkan struktur direktori dalam HTML
  - `get_system_info()`: Mendapatkan informasi sistem komprehensif
  - `install_requirements(requirements_file, additional_packages)`: Menginstal package yang diperlukan
  - `_detect_colab()`, `_detect_notebook()`: Deteksi lingkungan
- **Properties**:
  - `is_colab`: Cek apakah berjalan di Google Colab
  - `is_notebook`: Cek apakah berjalan di notebook
  - `base_dir`: Dapatkan direktori dasar
  - `drive_path`: Dapatkan path Google Drive
  - `is_drive_mounted`: Cek apakah Google Drive ter-mount
- **Fungsi Global**:
  - `get_environment_manager()`: Dapatkan singleton instance

### VisualizationBase (smartcash/common/visualization/core/visualization_base.py)
- **Fungsi**: Kelas dasar untuk semua komponen visualisasi
- **Metode Utama**:
  - `set_plot_style(style)`: Set style untuk matplotlib plots
  - `save_figure(fig, filepath, dpi)`: Simpan figure matplotlib
  - `create_output_directory(output_dir)`: Buat direktori output

### ChartHelper (smartcash/common/visualization/helpers/chart_helper.py)
- **Fungsi**: Visualisasi berbasis chart
- **Metode Utama**:
  - `create_bar_chart(data, title, horizontal, figsize, ...)`: Buat bar chart
  - `create_pie_chart(data, title, figsize, color_palette, ...)`: Buat pie chart
  - `create_line_chart(data, x_values, title, figsize, ...)`: Buat line chart
  - `create_heatmap(data, row_labels, col_labels, ...)`: Buat heatmap
  - `create_stacked_bar_chart(data, title, figsize, ...)`: Buat stacked bar chart

### ColorHelper (smartcash/common/visualization/helpers/color_helper.py)
- **Fungsi**: Manajemen warna untuk visualisasi
- **Metode Utama**:
  - `get_color_palette(n_colors, palette_name, as_hex, desat)`: Dapatkan palette warna
  - `create_color_mapping(categories, palette, as_hex)`: Buat mapping kategori ke warna
  - `generate_gradient(start_color, end_color, steps, as_hex)`: Buat gradien warna
  - `create_cmap(colors, name)`: Buat colormap kustom
  - `get_color_for_value(value, vmin, vmax, cmap_name, as_hex)`: Dapatkan warna berdasarkan nilai
  - `get_semantic_color(key, as_hex)`: Dapatkan warna semantik (success, warning, dll)

### AnnotationHelper (smartcash/common/visualization/helpers/annotation_helper.py)
- **Fungsi**: Anotasi pada visualisasi
- **Metode Utama**:
  - `add_bar_annotations(ax, bars, horizontal, fontsize, ...)`: Tambah anotasi pada bar chart
  - `add_stacked_bar_annotations(ax, bars, values, horizontal, ...)`: Tambah anotasi pada stacked bar
  - `add_line_annotations(ax, x_values, y_values, labels, ...)`: Tambah anotasi pada line chart
  - `create_legend(ax, labels, colors, title, ...)`: Buat legenda kustom
  - `add_data_labels(ax, x_values, y_values, labels, ...)`: Tambah label data
  - `get_pie_autopct_func(data, show_values, show_percents)`: Fungsi format untuk pie chart
  - `add_text_box(ax, text, x, y, fontsize, ...)`: Tambah text box
  - `add_annotated_heatmap(ax, data, text_format, threshold, ...)`: Tambah anotasi pada heatmap

### ExportHelper (smartcash/common/visualization/helpers/export_helper.py)
- **Fungsi**: Export visualisasi
- **Metode Utama**:
  - `save_figure(fig, output_path, dpi, format, ...)`: Simpan figure
  - `figure_to_base64(fig, format, dpi, close_fig)`: Konversi figure ke base64
  - `save_as_html(fig, output_path, title, include_plotlyjs, ...)`: Simpan sebagai HTML
  - `save_as_dashboard(figures, output_path, title, ...)`: Simpan multiple figures sebagai dashboard
  - `create_output_directory(output_dir)`: Buat direktori output

### LayoutHelper (smartcash/common/visualization/helpers/layout_helper.py)
- **Fungsi**: Layout untuk visualisasi
- **Metode Utama**:
  - `create_grid_layout(nrows, ncols, figsize, ...)`: Buat layout grid
  - `create_subplot_mosaic(mosaic, figsize, empty_sentinel)`: Buat layout dengan subplot_mosaic
  - `create_dashboard_layout(layout_spec, figsize, ...)`: Buat layout dashboard kustom
  - `adjust_subplots_spacing(fig, left, right, ...)`: Sesuaikan spacing subplots
  - `add_colorbar(fig, mappable, ax, location, ...)`: Tambahkan colorbar ke figure

### StyleHelper (smartcash/common/visualization/helpers/style_helper.py)
- **Fungsi**: Styling visualisasi
- **Metode Utama**:
  - `set_style(style, custom_params)`: Set style untuk visualisasi
  - `apply_style_to_figure(fig)`: Terapkan style ke figure
  - `apply_style_to_axes(ax)`: Terapkan style ke axes
  - `set_title_and_labels(ax, title, xlabel, ylabel)`: Set judul dan label dengan style
  - `get_current_style_params()`: Dapatkan parameter style saat ini
  - `register_custom_style(name, params)`: Daftarkan style kustom

### Constants (smartcash/common/constants.py)
- **Fungsi**: Konstanta global yang digunakan di seluruh project
- **Konstanta Utama**:
  - `VERSION`, `APP_NAME`: Metadata aplikasi
  - `DEFAULT_*_DIR`: Path direktori default
  - `DRIVE_BASE_PATH`: Path Google Drive untuk Colab
  - `DetectionLayer`: Enum untuk layer deteksi (BANKNOTE, NOMINAL, SECURITY)
  - `ModelFormat`: Enum format model (PYTORCH, ONNX, TORCHSCRIPT, dll)
  - `IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`: Ekstensi file yang didukung
  - `MODEL_EXTENSIONS`: Mapping format model ke ekstensi file
  - `ENV_*`: Environment variables
  - `DEFAULT_*`: Nilai default (confidence, IOU, img_size)
  - `MAX_BATCH_SIZE`, `MAX_IMAGE_SIZE`: Batasan aplikasi
  - `API_*`: Pengaturan API

### Utils (smartcash/common/utils.py)
- **Fungsi**: Berbagai utilitas umum untuk SmartCash
- **Fungsi Utama**:
  - `is_colab()`, `is_notebook()`: Deteksi lingkungan
  - `get_system_info()`: Dapatkan informasi sistem
  - `generate_unique_id()`: Generate ID unik
  - `format_time(seconds)`: Format waktu
  - `get_timestamp()`: Dapatkan timestamp untuk nama file
  - `ensure_dir(path)`: Pastikan direktori ada
  - File operations: `copy_file()`, `file_exists()`, `file_size()`, `format_size()`
  - Format operations: `load_json()`, `save_json()`, `load_yaml()`, `save_yaml()`
  - `get_project_root()`: Dapatkan root direktori project

### Interfaces (smartcash/common/interfaces/)
- **IDetectionVisualizer**: Interface untuk visualisasi deteksi
  - `visualize_detection(image, detections, filename, conf_threshold)`: Visualisasi hasil deteksi
- **IMetricsVisualizer**: Interface untuk visualisasi metrik
  - `plot_confusion_matrix(cm, class_names, title, filename)`: Plot confusion matrix
- **ILayerConfigManager**: Interface untuk konfigurasi layer
  - `get_layer_config(layer_name)`, `get_class_map()`, `get_layer_for_class_id(class_id)`
- **ICheckpointService**: Interface untuk checkpoint model
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`
  - `load_checkpoint(path, model, optimizer, map_location)`

## Exceptions (smartcash/common/exceptions.py)
- **Fungsi**: Hierarki exception terpadu untuk seluruh komponen SmartCash
- **Exception Classes Dasar**:
  - `SmartCashError`: Exception dasar untuk semua error SmartCash

- **Exception Config**:
  - `ConfigError`: Error konfigurasi

- **Exception Dataset**:
  - `DatasetError`: Error dasar terkait dataset
  - `DatasetFileError`: Error file dataset
  - `DatasetValidationError`: Error validasi dataset
  - `DatasetProcessingError`: Error pemrosesan dataset
  - `DatasetCompatibilityError`: Masalah kompatibilitas dataset dengan model

- **Exception Model**:
  - `ModelError`: Error dasar terkait model
  - `ModelConfigurationError`: Error konfigurasi model
  - `ModelTrainingError`: Error proses training model
  - `ModelInferenceError`: Error inferensi model
  - `ModelCheckpointError`: Error checkpoint model
  - `ModelExportError`: Error ekspor model
  - `ModelEvaluationError`: Error evaluasi model
  - `ModelServiceError`: Error model service

- **Exception Model Components**:
  - `ModelComponentError`: Error dasar komponen model
  - `BackboneError`: Error backbone model
  - `UnsupportedBackboneError`: Error backbone tidak didukung
  - `NeckError`: Error neck model
  - `HeadError`: Error detection head model

- **Exception Detection**:
  - `DetectionError`: Error dasar proses deteksi
  - `DetectionInferenceError`: Error inferensi deteksi
  - `DetectionPostprocessingError`: Error post-processing deteksi

- **Exception I/O**:
  - `FileError`: Error file I/O

- **Exception API & Validation**:
  - `APIError`: Error API
  - `ValidationError`: Error validasi input

- **Exception Lainnya**:
  - `NotSupportedError`: Fitur tidak didukung
  - `ExperimentError`: Error manajemen eksperimen

### Types (smartcash/common/types.py)
- **Fungsi**: Type definitions untuk SmartCash
- **Type Aliases**:
  - `ImageType`: Type untuk gambar (np.ndarray, string, bytes)
  - `PathType`: Type untuk path (string, Path)
  - `TensorType`: Type untuk tensor (torch.Tensor, np.ndarray)
  - `ConfigType`: Type untuk konfigurasi (Dict[str, Any])
  - `ProgressCallback`, `LogCallback`: Type untuk callback functions
- **Typed Dictionaries**:
  - `BoundingBox`: Bounding box dengan format [x1, y1, x2, y2]
  - `Detection`: Hasil deteksi objek
  - `ModelInfo`: Informasi model
  - `DatasetStats`: Statistik dataset

## DOMAIN COMPONENTS

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

## DOMAIN DATASET
# Pemetaan Lengkap Domain Dataset SmartCash

## DatasetManager (manager.py)
- **Fungsi**: Koordinator alur kerja dataset
- **Metode Utama**:
  - `__init__(config, logger)`: Inisialisasi dataset manager
  - `get_service(service_name)`: Lazy-initialization service dataset
  - `_get_preprocessor()`: Dapatkan preprocessor dataset dengan lazy initialization
  - `_get_preprocessed_loader()`: Dapatkan loader untuk dataset preprocessed
  - `preprocess_dataset(split, force_reprocess)`: Preprocess dataset dan simpan hasil
  - `clean_preprocessed(split)`: Bersihkan hasil preprocessing
  - `get_preprocessed_stats()`: Dapatkan statistik hasil preprocessing
  - `get_dataset(split, **kwargs)`: Dapatkan dataset untuk split tertentu
  - `get_dataloader(split, **kwargs)`: Dapatkan dataloader untuk split
  - `get_all_dataloaders(**kwargs)`: Dapatkan semua dataloader
  - `validate_dataset(split, **kwargs)`: Validasi dataset
  - `fix_dataset(split, **kwargs)`: Perbaiki masalah dataset
  - `augment_dataset(**kwargs)`: Augmentasi dataset
  - `download_from_roboflow(**kwargs)`: Download dataset dari Roboflow
  - `explore_class_distribution(split)`: Analisis distribusi kelas
  - `explore_layer_distribution(split)`: Analisis distribusi layer
  - `explore_bbox_statistics(split)`: Analisis statistik bbox
  - `balance_dataset(split, **kwargs)`: Seimbangkan dataset
  - `generate_dataset_report(splits, **kwargs)`: Buat laporan dataset
  - `visualize_class_distribution(class_stats, **kwargs)`: Visualisasi distribusi kelas
  - `create_dataset_dashboard(report, save_path)`: Buat dashboard visualisasi dataset
  - `get_split_statistics()`: Dapatkan statistik dasar split
  - `split_dataset(**kwargs)`: Pecah dataset menjadi train/val/test

## Dataset Services - Preprocessor

### DatasetPreprocessor (services/preprocessor/dataset_preprocessor.py)
- **Fungsi**: Layanan preprocessing dataset dan penyimpanan hasil untuk penggunaan berikutnya
- **Metode Utama**:
  - `__init__(config, logger)`: Inisialisasi preprocessor
  - `preprocess_dataset(split, force_reprocess, show_progress)`: Preprocess dataset dan simpan
  - `_preprocess_single_image(img_path, labels_dir, target_images_dir, target_labels_dir)`: Preprocess satu gambar
  - `clean_preprocessed(split)`: Bersihkan hasil preprocessing 
  - `get_preprocessed_stats()`: Dapatkan statistik hasil preprocessing
  - `is_preprocessed(split)`: Cek apakah split sudah dipreprocess

### PreprocessingPipeline (services/preprocessor/pipeline.py)
- **Fungsi**: Pipeline transformasi yang dapat dikonfigurasi untuk preprocessing
- **Metode Utama**:
  - `__init__(config, logger)`: Inisialisasi pipeline preprocessing
  - `setup_pipeline()`: Setup urutan transformasi
  - `process(image)`: Proses gambar melalui pipeline
  - `bgr_to_rgb(image)`: Konversi BGR ke RGB
  - `apply_letterbox(image)`: Resize dengan letterbox
  - `resize_direct(image)`: Resize langsung
  - `normalize(image)`: Normalisasi gambar
  - `create_training_pipeline(img_size)`: Factory pipeline training
  - `create_inference_pipeline(img_size)`: Factory pipeline inferensi

### PreprocessedStorage (services/preprocessor/storage.py)
- **Fungsi**: Pengelola penyimpanan hasil preprocessing dataset
- **Metode Utama**:
  - `__init__(base_dir, logger)`: Inisialisasi storage
  - `_load_metadata()`: Load metadata dari file
  - `_save_metadata()`: Simpan metadata ke file
  - `get_split_path(split)`: Dapatkan path untuk split
  - `get_split_metadata(split)`: Dapatkan metadata split
  - `update_split_metadata(split, metadata)`: Update metadata split
  - `save_preprocessed_image(split, image_id, image_data, metadata)`: Simpan gambar preprocessed
  - `load_preprocessed_image(split, image_id, with_metadata)`: Load gambar preprocessed
  - `copy_label_file(source_path, split, label_id)`: Salin file label
  - `list_preprocessed_images(split)`: Daftar gambar preprocessed
  - `get_stats(split)`: Dapatkan statistik storage
  - `update_stats(split, stats)`: Update statistik
  - `clean_storage(split)`: Bersihkan storage

### PreprocessedCleaner (services/preprocessor/cleaner.py)
- **Fungsi**: Pembersih cache dataset preprocessed
- **Metode Utama**:
  - `__init__(preprocessed_dir, max_age_days, logger)`: Inisialisasi cleaner
  - `clean_expired()`: Bersihkan data kadaluarsa
  - `clean_all()`: Bersihkan semua data
  - `clean_split(split)`: Bersihkan data untuk split
  - `_get_directory_size(directory)`: Hitung ukuran direktori

## Dataset Services - Loader

### DatasetLoaderService (services/loader/dataset_loader.py)
- **Fungsi**: Loading dataset dan pembuatan dataloader
- **Metode Utama**:
  - `get_dataset(split, transform, require_all_layers)`: Dapatkan dataset untuk split tertentu
  - `get_dataloader(split, batch_size, num_workers, shuffle)`: Dapatkan dataloader untuk split tertentu
  - `get_all_dataloaders(batch_size, num_workers)`: Dapatkan semua dataloader
  - `_get_split_path(split)`: Path untuk dataset split

### MultilayerLoader (services/loader/multilayer_loader.py)
- **Fungsi**: Loader khusus untuk dataset multilayer
- **Metode Utama**:
  - `get_dataset(split, transform, force_reload)`: Dapatkan dataset multilayer
  - `get_dataset_stats(split)`: Dapatkan statistik dataset
  - `clear_cache()`: Bersihkan cache dataset
  - `_get_split_path(split)`: Dapatkan path split

### PreprocessedDatasetLoader (services/loader/preprocessed_dataset_loader.py)
- **Fungsi**: Loader untuk dataset yang sudah dipreprocessing
- **Metode Utama**:
  - `__init__(preprocessed_dir, fallback_to_raw, auto_preprocess, config, logger)`: Inisialisasi loader
  - `get_dataset(split, require_all_layers, transform)`: Dapatkan dataset
  - `get_dataloader(split, batch_size, num_workers, shuffle, require_all_layers, transform)`: Dapatkan dataloader
  - `get_all_dataloaders(batch_size, num_workers, require_all_layers, transform)`: Dapatkan semua dataloader
  - `ensure_preprocessed(splits, force_reprocess)`: Pastikan dataset sudah dipreprocessing
  - `_is_split_available(split)`: Cek ketersediaan split
  - `_get_raw_dataset(split, require_all_layers, transform)`: Gunakan dataset raw

### BatchGenerator (services/loader/batch_generator.py)
- **Fungsi**: Generator batch optimasi dengan prefetching
- **Metode Utama**:
  - `__len__()`: Mendapatkan jumlah batch
  - `__iter__()`: Memulai iterasi batch

### DatasetCacheManager (services/loader/cache_manager.py)
- **Fungsi**: Manajemen cache dataset
- **Metode Utama**:
  - `get(key, default)`: Ambil item dari cache
  - `put(key, data, force)`: Simpan item ke cache
  - `clear(older_than_hours)`: Bersihkan cache
  - `get_stats()`: Dapatkan statistik cache

## Dataset Services - Validator

### DatasetValidatorService (services/validator/dataset_validator.py)
- **Fungsi**: Validasi dan perbaikan dataset
- **Metode Utama**:
  - `validate_dataset(split, fix_issues, move_invalid)`: Validasi dataset
  - `fix_dataset(split, fix_coordinates, fix_labels)`: Perbaiki dataset
  - `_validate_image_label_pair(img_path, labels_dir)`: Validasi pasangan gambar-label
  - `_aggregate_validation_results(results, stats)`: Agregasi hasil validasi
  - `_move_invalid_files(split, results)`: Pindahkan file tidak valid

### ImageValidator (services/validator/image_validator.py)
- **Fungsi**: Validasi gambar dataset
- **Metode Utama**:
  - `validate_image(image_path)`: Validasi satu file gambar
  - `fix_image(image_path)`: Perbaiki masalah pada gambar
  - `get_image_metadata(image_path)`: Dapatkan metadata gambar

### LabelValidator (services/validator/label_validator.py)
- **Fungsi**: Validasi file label dataset
- **Metode Utama**:
  - `validate_label(label_path, check_class_ids)`: Validasi file label
  - `fix_label(label_path, fix_coordinates, fix_format)`: Perbaiki file label
  - `check_layers_coverage(label_path, required_layers)`: Cek coverage layer

### DatasetFixer (services/validator/fixer.py)
- **Fungsi**: Memperbaiki dataset secara otomatis
- **Metode Utama**:
  - `fix_dataset(split, fix_images, fix_labels)`: Perbaiki dataset
  - `fix_orphaned_files(split, create_empty_labels)`: Perbaiki file orphaned
  - `_fix_image_label_pair(img_path, labels_dir)`: Perbaiki pasangan gambar-label

## Dataset Services - Augmentor

### AugmentationService (services/augmentor/augmentation_service.py)
- **Fungsi**: Layanan augmentasi dataset
- **Metode Utama**:
  - `augment_dataset(split, augmentation_types, target_count)`: Augmentasi dataset
  - `_get_augmentation_pipeline(augmentation_types)`: Dapatkan pipeline augmentasi
  - `_compute_augmentation_targets(class_distribution, target_count)`: Hitung target augmentasi
  - `_augment_class(source_files, pipeline, count)`: Augmentasi untuk satu kelas
  - `get_pipeline(augmentation_types)`: Dapatkan pipeline untuk penggunaan eksternal

### BBoxAugmentor (services/augmentor/bbox_augmentor.py)
- **Fungsi**: Augmentasi khusus bounding box
- **Metode Utama**:
  - `augment_bboxes(bboxes, class_ids, p, jitter, shift)`: Augmentasi bounding box
  - `mixup_bboxes(bboxes1, class_ids1, bboxes2, class_ids2)`: Gabungkan dua set bbox
  - `mosaic_bboxes(all_bboxes, all_class_ids, grid_size)`: Gabungkan multiple bbox
  - `_calculate_iou(bbox1, bbox2)`: Hitung IoU antar bbox

### ImageAugmentor (services/augmentor/image_augmentor.py)
- **Fungsi**: Augmentasi khusus gambar
- **Metode Utama**:
  - `cutout(image, p)`: Terapkan augmentasi cutout
  - `cutmix(image1, image2, ratio)`: Terapkan augmentasi cutmix
  - `mixup(image1, image2, alpha)`: Terapkan augmentasi mixup
  - `mosaic(images, grid_size)`: Terapkan augmentasi mosaic
  - `blend_with_gaussian_noise(image, p)`: Tambahkan noise
  - `random_erase(image, p)`: Terapkan random erase
  - `adjust_hue(image, p)`: Adjust hue gambar
  - `simulate_shadow(image, p)`: Simulasikan bayangan

### AugmentationPipelineFactory (services/augmentor/pipeline_factory.py)
- **Fungsi**: Factory untuk pipeline augmentasi
- **Metode Utama**:
  - `create_pipeline(augmentation_types, img_size)`: Buat pipeline custom
  - `create_train_pipeline(img_size)`: Buat pipeline untuk training
  - `create_validation_pipeline(img_size)`: Buat pipeline untuk validasi
  - `create_light_augmentation_pipeline(img_size)`: Buat pipeline augmentasi ringan
  - `create_heavy_augmentation_pipeline(img_size)`: Buat pipeline augmentasi berat

## Dataset Services - Explorer

### ExplorerService (services/explorer/explorer_service.py)
- **Fungsi**: Koordinator eksplorasi dataset
- **Metode Utama**:
  - `analyze_class_distribution(split, sample_size)`: Analisis distribusi kelas
  - `analyze_layer_distribution(split, sample_size)`: Analisis distribusi layer
  - `analyze_bbox_statistics(split, sample_size)`: Analisis statistik bbox
  - `analyze_image_sizes(split, sample_size)`: Analisis ukuran gambar

### BaseExplorer (services/explorer/base_explorer.py)
- **Fungsi**: Kelas dasar untuk semua explorer
- **Metode Utama**:
  - `_validate_directories(split)`: Validasi direktori dataset
  - `_get_valid_files(images_dir, labels_dir, sample_size)`: Dapatkan file valid

### BBoxExplorer (services/explorer/bbox_explorer.py)
- **Fungsi**: Analisis statistik bounding box
- **Metode Utama**:
  - `analyze_bbox_statistics(split, sample_size)`: Analisis statistik bbox
  - `_calc_stats(values)`: Hitung statistik dasar
  - `_categorize_areas(areas)`: Kategorikan area bbox

### ClassExplorer (services/explorer/class_explorer.py)
- **Fungsi**: Analisis distribusi kelas
- **Metode Utama**:
  - `analyze_distribution(split, sample_size)`: Analisis distribusi kelas

### DataExplorer (services/explorer/data_explorer.py)
- **Fungsi**: Eksplorasi dataset terpadu
- **Metode Utama**:
  - `explore_dataset(split, sample_size, output_dir)`: Eksplorasi komprehensif
  - `_generate_visualizations(results, split, output_dir)`: Generate visualisasi
  - `_generate_insights(results)`: Generate insight dari hasil analisis
  - `export_analysis_report(results, output_format)`: Export laporan analisis

### ImageExplorer (services/explorer/image_explorer.py)
- **Fungsi**: Analisis ukuran dan properti gambar
- **Metode Utama**:
  - `analyze_image_sizes(split, sample_size)`: Analisis ukuran gambar

### LayerExplorer (services/explorer/layer_explorer.py)
- **Fungsi**: Analisis distribusi layer
- **Metode Utama**:
  - `analyze_distribution(split, sample_size)`: Analisis distribusi layer

## Dataset Services - Balancer

### BalanceService (services/balancer/balance_service.py)
- **Fungsi**: Menyeimbangkan dataset
- **Metode Utama**:
  - `balance_by_undersampling(split, strategy, target_count)`: Undersampling dataset
  - `balance_by_oversampling(split, target_count, augmentation_types)`: Oversampling dataset
  - `calculate_weights(split)`: Hitung bobot sampling untuk weighted dataset
  - `_analyze_class_distribution(images_dir, labels_dir)`: Analisis distribusi kelas
  - `_copy_file_pair(img_path, label_path, output_dir)`: Salin pasangan file

### Oversampler (services/balancer/oversampler.py)
- **Fungsi**: Metode oversampling dataset
- **Metode Utama**:
  - `oversample(data, labels, strategy, target_count)`: Lakukan oversampling
  - `_generate_by_strategy(data, indices, strategy, n_samples)`: Generate sampel
  - `_duplicate_samples(data, indices, n_samples)`: Duplikasi sampel
  - `_smote_samples(data, indices, n_samples)`: Generate dengan SMOTE
  - `_adasyn_samples(data, indices, n_samples)`: Generate dengan ADASYN
  - `_augmentation_samples(data, indices, n_samples, pipeline)`: Generate dengan augmentasi

### Undersampler (services/balancer/undersampler.py)
- **Fungsi**: Metode undersampling dataset
- **Metode Utama**:
  - `undersample(data, labels, strategy, target_count)`: Lakukan undersampling
  - `_select_by_strategy(data, indices, strategy, target_count)`: Pilih sampel
  - `_cluster_undersampling(data, indices, target_count)`: Cluster undersampling
  - `_neighbour_undersampling(data, indices, target_count)`: Nearest neighbour
  - `_tomek_undersampling(data, indices, target_count)`: Tomek links

### WeightCalculator (services/balancer/weight_calculator.py)
- **Fungsi**: Hitung bobot sampling
- **Metode Utama**:
  - `calculate_class_weights(class_counts, strategy, beta)`: Hitung bobot kelas
  - `calculate_sample_weights(targets, class_weights)`: Hitung bobot sampel
  - `get_balanced_sampler_weights(dataset, strategy)`: Bobot untuk weighted sampler
  - `calculate_focal_loss_weights(class_counts, gamma, alpha)`: Bobot focal loss
  - `get_label_smoothing_factor(class_counts)`: Faktor label smoothing

## Dataset Services - Reporter

### ReportService (services/reporter/report_service.py)
- **Fungsi**: Layanan untuk membuat laporan dataset
- **Metode Utama**:
  - `generate_dataset_report(splits, visualize, calculate_metrics)`: Buat laporan
  - `_analyze_splits_parallel(splits)`: Analisis semua split secara paralel
  - `_analyze_split(split)`: Analisis satu split dataset
  - `_compile_metrics(report)`: Kompilasi metrik dari hasil analisis
  - `_generate_visualizations(report, sample_count)`: Generate visualisasi
  - `_calculate_quality_score(report)`: Hitung skor kualitas dataset
  - `_generate_recommendations(report)`: Generate rekomendasi
  - `export_report(report, formats)`: Ekspor laporan dalam berbagai format
  - `generate_comparison_report(reports, labels)`: Laporan perbandingan dataset

## Dataset Components

### PreprocessedMultilayerDataset (services/loader/preprocessed_dataset_loader.py)
- **Fungsi**: Dataset untuk data yang sudah dipreprocessing dengan dukungan multilayer
- **Metode Utama**:
  - `__init__(root_dir, img_size, require_all_layers, transform, logger)`: Inisialisasi dataset
  - `__len__()`: Dapatkan jumlah item
  - `__getitem__(idx)`: Dapatkan satu item

### MultilayerDataset (components/datasets/multilayer_dataset.py)
- **Fungsi**: Dataset multilayer untuk deteksi
- **Metode Utama**:
  - `__getitem__(idx)`: Ambil item dataset
  - `get_layer_annotation(img_id, layer)`: Anotasi per layer

## Dataset Utils - Transformasi

### ImageTransformer (utils/transform/image_transform.py)
- **Fungsi**: Transformasi dan augmentasi gambar
- **Metode Utama**:
  - `_setup_transformations()`: Setup pipeline transformasi
  - `get_transform(mode)`: Dapatkan transformasi sesuai mode
  - `create_custom_transform(**kwargs)`: Buat transformasi kustom
  - `process_image(image, bboxes, class_labels, mode)`: Proses gambar
  - `get_normalization_params()`: Dapatkan parameter normalisasi
  - `resize_image(image, keep_ratio)`: Resize gambar ke target size

### BBoxTransformer (utils/transform/bbox_transform.py)
- **Fungsi**: Transformasi dan konversi format bbox
- **Metode Utama**:
  - `yolo_to_xyxy(bbox, img_width, img_height)`: Konversi YOLO ke XYXY
  - `xyxy_to_yolo(bbox, img_width, img_height)`: Konversi XYXY ke YOLO
  - `yolo_to_coco(bbox, img_width, img_height)`: Konversi YOLO ke COCO
  - `coco_to_yolo(bbox, img_width, img_height)`: Konversi COCO ke YOLO
  - `clip_bbox(bbox, format)`: Clip bbox ke range valid
  - `scale_bbox(bbox, scale_factor, format)`: Skala ukuran bbox
  - `iou(boxA, boxB, format)`: Hitung IoU antara dua bbox

### PolygonTransformer (utils/transform/polygon_transform.py)
- **Fungsi**: Transformasi dan konversi format polygon
- **Metode Utama**:
  - `normalize_polygon(polygon, img_width, img_height)`: Normalisasi koordinat
  - `denormalize_polygon(polygon, img_width, img_height)`: Denormalisasi koordinat
  - `polygon_to_binary_mask(polygon, img_width, img_height)`: Konversi ke mask
  - `binary_mask_to_polygon(mask, epsilon, normalize)`: Konversi mask ke polygon
  - `find_bbox_from_polygon(polygon, img_width, img_height)`: Temukan bbox
  - `simplify_polygon(polygon, epsilon, img_width, img_height)`: Sederhanakan polygon

### FormatConverter (utils/transform/format_converter.py)
- **Fungsi**: Konversi format dataset
- **Metode Utama**:
  - `yolo_to_coco(source_dir, target_path, class_map)`: Konversi YOLO ke COCO
  - `coco_to_yolo(source_path, target_dir, images_src)`: Konversi COCO ke YOLO
  - `yolo_to_voc(source_dir, target_dir, class_map)`: Konversi YOLO ke VOC
  - `voc_to_yolo(source_dir, target_dir, class_map)`: Konversi VOC ke YOLO

### AlbumentationsAdapter (utils/transform/albumentations_adapter.py)
- **Fungsi**: Adapter untuk library Albumentations
- **Metode Utama**:
  - `get_basic_transforms(img_size, normalize)`: Dapatkan transformasi dasar
  - `get_geometric_augmentations(img_size, p, scale)`: Augmentasi geometrik
  - `get_color_augmentations(p, brightness_limit)`: Augmentasi warna
  - `get_noise_augmentations(p, with_bbox)`: Augmentasi noise
  - `create_augmentation_pipeline(img_size, types)`: Buat pipeline augmentasi
  - `apply_transforms(image, bboxes, class_labels)`: Terapkan transformasi

## Dataset Utils - Split

### DatasetSplitter (utils/split/dataset_splitter.py)
- **Fungsi**: Memecah dataset menjadi train/val/test
- **Metode Utama**:
  - `split_dataset(train_ratio, val_ratio, test_ratio)`: Pecah dataset
  - `_detect_data_structure(directory)`: Deteksi struktur dataset
  - `_split_flat_dataset(source_dir, train_ratio, val_ratio)`: Pecah dataset flat
  - `_copy_files(split, files, use_symlinks)`: Salin file ke split
  - `_count_existing_splits(directory)`: Hitung jumlah file dalam split

### DatasetMerger (utils/split/merger.py)
- **Fungsi**: Menggabungkan beberapa dataset menjadi satu
- **Metode Utama**:
  - `merge_datasets(source_dirs, output_dir, prefix_filenames)`: Gabung dataset
  - `merge_splits(source_dir, output_dir, splits_to_merge)`: Gabung beberapa split

### DatasetStratifier (utils/split/stratifier.py)
- **Fungsi**: Stratifikasi dataset berdasarkan kriteria
- **Metode Utama**:
  - `stratify_by_class(files, class_ratios, random_seed)`: Stratifikasi per kelas
  - `stratify_by_layer(files, layer_ratios, random_seed)`: Stratifikasi per layer
  - `stratify_by_count(files, train_count, valid_count)`: Stratifikasi per jumlah
  - `_group_files_by_class(files)`: Kelompokkan file berdasarkan kelas
  - `_group_files_by_layer(files)`: Kelompokkan file berdasarkan layer

## Dataset Utils - Statistik dan File

### ClassStatistics (utils/statistics/class_stats.py)
- **Fungsi**: Analisis statistik distribusi kelas
- **Metode Utama**:
  - `analyze_distribution(split, sample_size)`: Analisis distribusi kelas
  - `get_class_weights(split, method)`: Hitung bobot kelas untuk imbalance
  - `_calculate_distribution(image_files, labels_dir)`: Hitung distribusi

### ImageStatistics (utils/statistics/image_stats.py)
- **Fungsi**: Analisis statistik gambar
- **Metode Utama**:
  - `analyze_image_sizes(split, sample_size)`: Analisis ukuran gambar
  - `analyze_image_quality(split, sample_size)`: Analisis kualitas gambar
  - `find_problematic_images(split, threshold)`: Temukan gambar bermasalah

### DistributionAnalyzer (utils/statistics/distribution_analyzer.py)
- **Fungsi**: Analisis distribusi statistik dataset
- **Metode Utama**:
  - `analyze_dataset(splits, sample_size)`: Analisis komprehensif dataset
  - `_analyze_cross_split_consistency(results)`: Analisis konsistensi antar split
  - `_generate_suggestions(results)`: Buat rekomendasi dari hasil analisis

### FileProcessor (utils/file/file_processor.py)
- **Fungsi**: Pemrosesan file dan direktori
- **Metode Utama**:
  - `count_files(directory, extensions)`: Hitung jumlah file
  - `copy_files(source_dir, target_dir, file_list)`: Salin file
  - `move_files(source_dir, target_dir, file_list)`: Pindahkan file
  - `extract_zip(zip_path, output_dir, include_patterns)`: Ekstrak zip

### ImageProcessor (utils/file/image_processor.py)
- **Fungsi**: Pemrosesan dan manipulasi gambar
- **Metode Utama**:
  - `resize_images(directory, target_size, output_dir)`: Resize gambar
  - `enhance_images(directory, output_dir, enhance_contrast)`: Tingkatkan kualitas
  - `convert_format(directory, target_format, output_dir)`: Konversi format

### LabelProcessor (utils/file/label_processor.py)
- **Fungsi**: Pemrosesan dan manipulasi file label
- **Metode Utama**:
  - `fix_labels(directory, fix_coordinates, fix_class_ids)`: Perbaiki label
  - `filter_classes(directory, keep_classes, remove_classes)`: Filter kelas
  - `extract_layer(directory, layer_name, output_dir)`: Ekstrak satu layer

## Dataset Utils - Progress Tracking

### ProgressTracker (utils/progress/progress_tracker.py)
- **Fungsi**: Tracking progres operasi dataset
- **Metode Utama**:
  - `update(n, message)`: Update progres
  - `set_total(total)`: Set total unit
  - `add_callback(callback)`: Tambah callback function
  - `add_subtask(subtask)`: Tambah subtask ke tracker
  - `set_metrics(metrics)`: Set metrik pelacakan
  - `get_progress()`: Dapatkan status progres
  - `complete(message)`: Tandai progres selesai

### ProgressObserver (utils/progress/observer_adapter.py)
- **Fungsi**: Observer untuk event progress
- **Metode Utama**:
  - `update(event_type, sender, **kwargs)`: Update status dari event

### ProgressEventEmitter (utils/progress/observer_adapter.py)
- **Fungsi**: Emitter untuk event progress
- **Metode Utama**:
  - `start(description, total)`: Mulai progres dan kirim event
  - `update(progress, message, metrics)`: Update progres
  - `increment(increment, message, metrics)`: Tambah nilai progres
  - `complete(message, final_metrics)`: Tandai selesai dan kirim event

## Dataset Visualization

### DataVisualizationHelper (visualization/data.py)
- **Fungsi**: Visualisasi data dan dataset
- **Metode Utama**:
  - `plot_class_distribution(class_stats, title, save_path)`: Visualisasi kelas
  - `plot_layer_distribution(layer_stats, title, save_path)`: Visualisasi layer
  - `plot_sample_images(data_dir, num_samples, classes)`: Visualisasi sampel
  - `plot_augmentation_comparison(image_path, types)`: Visualisasi augmentasi

### ReportVisualizer (visualization/report.py)
- **Fungsi**: Visualisasi laporan dataset
- **Metode Utama**:
  - `create_class_distribution_summary(class_stats, title)`: Visualisasi kelas
  - `create_dataset_dashboard(report, save_path)`: Buat dashboard visualisasi
  - `_plot_split_distribution(ax, report)`: Plot distribusi split
  - `_plot_class_summary(ax, report)`: Plot ringkasan kelas
  - `_plot_layer_distribution(ax, report)`: Plot distribusi layer
  - `_plot_bbox_distribution(ax, report)`: Plot distribusi bbox
  - `_plot_quality_metrics(ax, report)`: Plot metrik kualitas
  - `_plot_recommendations(ax, report)`: Plot rekomendasi

### ReportService (services/reporter/report_service.py)
- **Fungsi**: Layanan untuk membuat laporan dataset
- **Metode Utama**:
  - `generate_dataset_report(splits, visualize, calculate_metrics)`: Buat laporan
  - `_analyze_splits_parallel(splits)`: Analisis semua split secara paralel
  - `_analyze_split(split)`: Analisis satu split dataset
  - `_compile_metrics(report)`: Kompilasi metrik dari hasil analisis
  - `_generate_visualizations(report, sample_count)`: Generate visualisasi
  - `_calculate_quality_score(report)`: Hitung skor kualitas dataset
  - `_generate_recommendations(report)`: Generate rekomendasi
  - `export_report(report, formats)`: Ekspor laporan dalam berbagai format
  - `generate_comparison_report(reports, labels)`: Laporan perbandingan dataset

## DOMAIN DETECTION

### Detector (detector.py)
- **Fungsi**: Koordinator utama proses deteksi
- **Metode Utama**:
  - `__init__(model_manager, prediction_service, inference_service, postprocessing_service, visualization_adapter, logger)`: Inisialisasi detector dengan dependensi
  - `detect(image, conf_threshold, iou_threshold, with_visualization)`: Deteksi pada gambar tunggal dengan opsi visualisasi
  - `detect_multilayer(image, threshold)`: Deteksi multilayer dengan threshold khusus per layer
  - `detect_batch(images, conf_threshold, iou_threshold)`: Deteksi pada batch gambar
  - `visualize(image, detections, conf_threshold, show_labels, show_conf, filename)`: Visualisasi hasil deteksi

### DetectionVisualizationAdapter (services/visualization_adapter.py)
- **Fungsi**: Adapter visualisasi dari domain model ke domain detection
- **Metode Utama**:
  - `__init__(detection_output_dir, metrics_output_dir, logger)`: Inisialisasi adapter visualisasi
  - `visualize_detection(image, detections, filename, conf_threshold, show_labels, show_conf)`: Visualisasi hasil deteksi
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize)`: Visualisasi confusion matrix
  - `visualize_model_comparison(comparison_data, metric_cols, title, filename)`: Visualisasi perbandingan model

### InferenceService (services/inference/inference_service.py)
- **Fungsi**: Koordinator layanan inferensi model
- **Metode Utama**:
  - `__init__(prediction_service, postprocessing_service, accelerator, logger)`: Inisialisasi layanan dengan dependensi
  - `infer(image, conf_threshold, iou_threshold)`: Inferensi pada gambar tunggal
  - `batch_infer(images, conf_threshold, iou_threshold)`: Inferensi pada batch gambar
  - `visualize(image, detections)`: Visualisasi hasil inferensi
  - `optimize_model(target_format, **kwargs)`: Optimasi model untuk inferensi
  - `get_model_info()`: Informasi model yang digunakan

### HardwareAccelerator (services/inference/accelerator.py)
- **Fungsi**: Abstraksi hardware untuk akselerasi inferensi
- **Metode Utama**:
  - `__init__(accelerator_type, device_id, use_fp16, logger)`: Inisialisasi akselerator
  - `setup()`: Setup akselerator untuk inferensi
  - `get_device()`: Dapatkan device yang dikonfigurasi
  - `get_device_info()`: Informasi device (CUDA, MPS, CPU, etc)
  - `_auto_detect_hardware()`: Deteksi otomatis hardware terbaik
  - `_setup_cuda/mps/tpu/rocm/cpu()`: Setup untuk berbagai tipe akselerator

### BatchProcessor (services/inference/batch_processor.py)
- **Fungsi**: Processor untuk inferensi batch gambar paralel
- **Metode Utama**:
  - `__init__(inference_service, output_dir, num_workers, batch_size, logger)`: Inisialisasi processor
  - `process_directory(input_dir, output_dir, extensions, recursive, conf_threshold, iou_threshold, save_results, save_visualizations, result_format, callback)`: Proses semua gambar dalam direktori
  - `process_batch(images, output_dir, filenames, conf_threshold, iou_threshold, save_results, save_visualizations, result_format)`: Proses batch gambar yang sudah dimuat
  - `_save_result(img_path, detections, output_path, save_visualization, format)`: Simpan hasil deteksi

### ModelOptimizer (services/inference/optimizers.py)
- **Fungsi**: Utilitas optimasi model untuk inferensi
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi optimizer
  - `optimize_to_onnx(model, output_path, input_shape, dynamic_axes, opset_version, simplify)`: Optimasi ke ONNX
  - `optimize_to_torchscript(model, output_path, input_shape, method)`: Optimasi ke TorchScript
  - `optimize_to_tensorrt(onnx_path, output_path, fp16_mode, int8_mode, workspace_size)`: Optimasi ke TensorRT
  - `optimize_to_tflite(model_path, output_path, quantize, input_shape)`: Optimasi ke TFLite
  - `optimize_model(model, model_format, output_path, **kwargs)`: Optimasi model ke format yang ditentukan

### PostprocessingService (services/postprocessing/postprocessing_service.py)
- **Fungsi**: Koordinator postprocessing hasil deteksi
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi service
  - `process(detections, conf_threshold, iou_threshold, refine_boxes, class_specific_nms, max_detections)`: Proses postprocessing lengkap

### ConfidenceFilter (services/postprocessing/confidence_filter.py)
- **Fungsi**: Filter deteksi berdasarkan confidence threshold
- **Metode Utama**:
  - `__init__(default_threshold, class_thresholds, logger)`: Inisialisasi filter
  - `process(detections, global_threshold)`: Filter deteksi berdasarkan threshold
  - `set_threshold(class_id, threshold)`: Set threshold per kelas
  - `get_threshold(class_id)`: Dapatkan threshold untuk kelas
  - `reset_thresholds()`: Reset semua threshold ke default

### BBoxRefiner (services/postprocessing/bbox_refiner.py)
- **Fungsi**: Perbaikan bounding box hasil deteksi
- **Metode Utama**:
  - `__init__(clip_boxes, expand_factor, logger)`: Inisialisasi refiner
  - `process(detections, image_width, image_height, specific_classes)`: Perbaiki bounding box deteksi
  - `_expand_bbox(bbox, factor)`: Ekspansi bbox dengan factor tertentu
  - `_clip_bbox(bbox)`: Clip bbox ke range [0,1]
  - `_fix_absolute_bbox(bbox, img_width, img_height)`: Perbaiki bbox dalam koordinat absolut

### ResultFormatter (services/postprocessing/result_formatter.py)
- **Fungsi**: Format hasil deteksi ke berbagai format
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi formatter
  - `to_json(detections, include_metadata, pretty)`: Format ke JSON
  - `to_csv(detections, include_header)`: Format ke CSV
  - `to_yolo_format(detections)`: Format ke format YOLO
  - `to_coco_format(detections, image_id, image_width, image_height)`: Format ke COCO
  - `format_detections(detections, format, **kwargs)`: Format dengan format yang ditentukan

### DetectionHandler (handlers/detection_handler.py)
- **Fungsi**: Handler untuk deteksi gambar tunggal
- **Metode Utama**:
  - `__init__(inference_service, postprocessing_service, logger)`: Inisialisasi handler
  - `detect(image, conf_threshold, iou_threshold, apply_postprocessing, return_visualization)`: Deteksi objek pada gambar
  - `save_result(detections, output_path, image, save_visualization, format)`: Simpan hasil deteksi ke file

### BatchHandler (handlers/batch_handler.py)
- **Fungsi**: Handler untuk deteksi batch/kumpulan gambar
- **Metode Utama**:
  - `__init__(detection_handler, num_workers, batch_size, max_batch_size, logger)`: Inisialisasi handler batch
  - `detect_directory(input_dir, output_dir, extensions, recursive, conf_threshold, iou_threshold, save_results, save_visualizations, result_format)`: Deteksi pada semua gambar di direktori
  - `detect_zip(zip_path, output_dir, conf_threshold, iou_threshold, extensions, save_extracted)`: Deteksi pada gambar dalam file ZIP

### VideoHandler (handlers/video_handler.py)
- **Fungsi**: Handler untuk deteksi video dan webcam
- **Metode Utama**:
  - `__init__(detection_handler, logger)`: Inisialisasi handler
  - `detect_video(video_path, output_path, conf_threshold, iou_threshold, start_frame, end_frame, step, show_progress, show_preview, overlay_info, callback)`: Deteksi pada file video
  - `detect_webcam(camera_id, output_path, conf_threshold, iou_threshold, display_width, display_height, overlay_info, max_time, callback)`: Deteksi pada webcam
  - `stop()`: Hentikan proses yang berjalan
  - `_add_overlay_text(frame, text_lines, start_y, line_height, color, thickness, font_scale)`: Tambahkan teks overlay

### IntegrationHandler (handlers/integration_handler.py)
- **Fungsi**: Handler untuk integrasi dengan UI/API
- **Metode Utama**:
  - `__init__(detection_handler, logger)`: Inisialisasi handler
  - `detect_from_base64(image_base64, conf_threshold, iou_threshold, return_visualization, visualization_format)`: Deteksi dari gambar base64
  - `start_async_worker(num_workers)`: Mulai worker thread asinkron
  - `stop_async_worker()`: Hentikan worker thread
  - `detect_async(image, task_id, conf_threshold, iou_threshold, callback, return_visualization)`: Deteksi asinkron
  - `get_task_result(task_id, remove_after_get)`: Dapatkan hasil task asinkron
  - `get_queue_status()`: Status queue dan task
  - `clean_old_results(max_age_seconds)`: Bersihkan hasil lama
  - `_async_worker_loop()`: Loop worker asinkron
  - `to_json_response(result)`: Konversi hasil ke respons JSON

### ONNXModelAdapter (adapters/onnx_adapter.py)
- **Fungsi**: Adapter untuk model ONNX yang dioptimasi
- **Metode Utama**:
  - `__init__(onnx_path, input_shape, class_map, logger)`: Inisialisasi adapter
  - `_load_model()`: Load model ONNX
  - `preprocess(image)`: Preprocess gambar untuk inferensi
  - `postprocess(outputs, original_image, conf_threshold, iou_threshold)`: Postprocess hasil
  - `predict(image, conf_threshold, iou_threshold)`: Prediksi objek pada gambar
  - `visualize(image, detections, conf_threshold)`: Visualisasi hasil

### TorchScriptAdapter (adapters/torchscript_adapter.py)
- **Fungsi**: Adapter untuk model TorchScript yang dioptimasi
- **Metode Utama**:
  - `__init__(model_path, input_shape, class_map, device, logger)`: Inisialisasi adapter
  - `_load_model()`: Load model TorchScript
  - `preprocess(image)`: Preprocess gambar untuk inferensi
  - `postprocess(output, original_image, conf_threshold, iou_threshold)`: Postprocess hasil
  - `predict(image, conf_threshold, iou_threshold)`: Prediksi objek pada gambar
  - `visualize(image, detections, conf_threshold)`: Visualisasi hasil

## DOMAIN MODEL

## ModelManager (manager.py)
- **Fungsi**: Koordinator alur kerja model
- **Metode Utama**:
  - `__init__(config, model_type, logger)`: Inisialisasi Model Manager
  - `_validate_config()`: Validasi konfigurasi model
  - `build_model()`: Buat dan inisialisasi model
  - `_build_backbone()`: Buat backbone berdasarkan konfigurasi
  - `_build_neck()`: Buat neck berdasarkan konfigurasi
  - `_build_head()`: Buat detection head berdasarkan konfigurasi
  - `_build_loss_function()`: Buat loss function berdasarkan konfigurasi
  - `create_model(model_type, **kwargs)`: Factory method untuk membuat model berdasarkan tipe
  - `get_config()`: Dapatkan konfigurasi model
  - `update_config(config_updates)`: Update konfigurasi model
  - `set_checkpoint_service(service)`: Set checkpoint service untuk model manager
  - `set_training_service(service)`: Set training service untuk model manager
  - `set_evaluation_service(service)`: Set evaluation service untuk model manager
  - `save_model(path)`: Simpan model ke file
  - `load_model(path)`: Load model dari file
  - `train(*args, **kwargs)`: Latih model
  - `evaluate(*args, **kwargs)`: Evaluasi model
  - `predict(image, conf_threshold, iou_threshold)`: Lakukan prediksi pada gambar

## ModelCheckpointManager (manager_checkpoint.py)
- **Fungsi**: Integrasi checkpoint service dengan model manager
- **Metode Utama**:
  - `__init__(model_manager, checkpoint_dir, max_checkpoints, logger)`: Inisialisasi ModelCheckpointManager
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint model
  - `load_checkpoint(path, model, optimizer, map_location)`: Load checkpoint ke model
  - `get_best_checkpoint()`: Dapatkan path ke checkpoint terbaik
  - `get_latest_checkpoint()`: Dapatkan path ke checkpoint terbaru
  - `list_checkpoints(sort_by)`: Daftar semua checkpoint yang tersedia
  - `export_to_onnx(output_path, input_shape, opset_version, dynamic_axes)`: Export model ke ONNX

## Visualisasi

### ModelVisualizationBase (visualization/base_visualizer.py)
- **Fungsi**: Kelas dasar untuk visualisasi model
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi visualizer
  - `set_plot_style(style)`: Set style untuk matplotlib plots
  - `save_figure(fig, filepath, dpi, bbox_inches)`: Simpan figure matplotlib
  - `create_output_directory(output_dir)`: Buat direktori output

### MetricsVisualizer (visualization/metrics_visualizer.py)
- **Fungsi**: Visualisasi metrik evaluasi model
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi metrics visualizer
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize, figsize, cmap)`: Plot confusion matrix
  - `plot_training_metrics(metrics, title, filename, figsize, include_lr)`: Plot metrik training
  - `plot_accuracy_metrics(metrics, title, filename, figsize)`: Plot metrik akurasi
  - `plot_model_comparison(comparison_data, metric_cols, title, filename, figsize)`: Plot perbandingan metrik model
  - `plot_research_comparison(results_df, metric_cols, title, filename, figsize)`: Plot perbandingan hasil skenario

### DetectionVisualizer (visualization/detection_visualizer.py)
- **Fungsi**: Visualisasi hasil deteksi objek
- **Metode Utama**:
  - `__init__(output_dir, class_colors, logger)`: Inisialisasi visualizer deteksi
  - `visualize_detection(image, detections, filename, conf_threshold, show_labels, show_conf, show_total, show_value)`: Visualisasikan deteksi pada gambar
  - `visualize_detections_grid(images, detections_list, title, filename, grid_size, conf_threshold)`: Visualisasikan multiple deteksi dalam grid
  - `calculate_denomination_total(detections)`: Hitung total nilai mata uang dari deteksi
  - `_create_grid(images, grid_size, title)`: Buat grid dari gambar

### EvaluationVisualizer (visualization/evaluation_visualizer.py)
- **Fungsi**: Visualisasi hasil evaluasi model
- **Metode Utama**:
  - `__init__(config, output_dir, logger)`: Inisialisasi visualizer evaluasi
  - `create_all_plots(metrics_data, prefix, **kwargs)`: Buat semua visualisasi yang tersedia
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize, **kwargs)`: Plot confusion matrix
  - `plot_map_f1_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan mAP dan F1
  - `plot_inference_time(metrics_df, prefix, **kwargs)`: Plot perbandingan waktu inferensi
  - `plot_backbone_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan backbone
  - `plot_condition_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan kondisi pengujian
  - `plot_combined_heatmap(metrics_df, prefix, **kwargs)`: Plot heatmap kombinasi
  - `visualize_predictions(samples, conf_thres, title, filename, max_samples, save_path)`: Visualisasi hasil prediksi model
  - `plot_metrics_history(metrics_history, title, figsize, save_path)`: Visualisasi history metrik training/validasi

### BaseResearchVisualizer (visualization/research/base_research_visualizer.py)
- **Fungsi**: Kelas dasar untuk visualisasi hasil penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi base visualizer penelitian
  - `_create_styled_dataframe(df)`: Buat DataFrame dengan styling untuk highlight nilai terbaik
  - `_add_tradeoff_regions(ax)`: Tambahkan regions untuk visualisasi trade-off
  - `save_visualization(fig, filename)`: Simpan visualisasi ke file

### ExperimentVisualizer (visualization/research/experiment_visualizer.py)
- **Fungsi**: Visualisasi hasil eksperimen model
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi experiment visualizer
  - `visualize_experiment_comparison(results_df, title, filename, highlight_best, figsize)`: Visualisasi perbandingan hasil eksperimen
  - `_create_backbone_based_plots(axes, results_df, metric_cols, time_col)`: Buat plot berdasarkan backbone
  - `_create_general_plots(axes, results_df, metric_cols, time_col)`: Buat plot umum

### ScenarioVisualizer (visualization/research/scenario_visualizer.py)
- **Fungsi**: Visualisasi hasil skenario penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi scenario visualizer
  - `visualize_scenario_comparison(results_df, title, filename, figsize)`: Visualisasi perbandingan skenario penelitian
  - `_filter_successful_scenarios(df)`: Filter skenario yang sukses
  - `_add_scenario_column(df)`: Tambahkan kolom Skenario
  - `_identify_columns(df)`: Identifikasi kolom-kolom penting
  - `_create_accuracy_plot(ax, df, metric_cols, backbone_col)`: Buat plot akurasi per skenario
  - `_create_inference_time_plot(ax, df, time_col, backbone_col)`: Buat plot waktu inferensi
  - `_create_backbone_comparison_plot(ax, df, metric_cols, backbone_col)`: Buat plot perbandingan backbone
  - `_create_condition_comparison_plot(ax, df, metric_cols, condition_col)`: Buat plot perbandingan kondisi

### ResearchVisualizer (visualization/research/research_visualizer.py)
- **Fungsi**: Visualisasi dan analisis hasil penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi research visualizer
  - `visualize_experiment_comparison(results_df, title, filename, highlight_best, figsize)`: Visualisasi perbandingan eksperimen
  - `visualize_scenario_comparison(results_df, title, filename, figsize)`: Visualisasi perbandingan skenario

## Services

### CheckpointService (services/checkpoint/checkpoint_service.py)
- **Fungsi**: Layanan untuk mengelola checkpoint model
- **Metode Utama**:
  - `__init__(checkpoint_dir, max_checkpoints, logger)`: Inisialisasi Checkpoint Service
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint
  - `load_checkpoint(path, model, optimizer, map_location)`: Load checkpoint
  - `get_latest_checkpoint()`: Dapatkan path ke checkpoint terbaru
  - `get_best_checkpoint()`: Dapatkan path ke checkpoint terbaik
  - `list_checkpoints(sort_by)`: Daftar checkpoint yang tersedia
  - `export_to_onnx(model, output_path, input_shape, opset_version, dynamic_axes)`: Export ke ONNX
  - `add_metadata(checkpoint_path, metadata)`: Tambahkan metadata ke checkpoint
  - `_cleanup_old_checkpoints()`: Hapus checkpoint lama

### TrainingService (services/training/core_training_service.py)
- **Fungsi**: Layanan pelatihan model
- **Metode Utama**:
  - `__init__(model, config, device, logger, experiment_name)`: Inisialisasi training service
  - `_setup_components()`: Setup komponen training
  - `_setup_experiment_tracker()`: Setup experiment tracker
  - `train(train_loader, val_loader, callbacks)`: Training model
  - `_train_epoch(train_loader)`: Proses training untuk satu epoch
  - `_validate_epoch(val_loader)`: Proses validasi untuk satu epoch
  - `_compute_loss(outputs, targets)`: Hitung loss untuk prediksi dan target
  - `_compute_metrics(outputs, targets)`: Hitung metrics selain loss
  - `_save_checkpoint(checkpoint_name)`: Simpan checkpoint model
  - `resume_from_checkpoint(checkpoint_path)`: Lanjutkan training dari checkpoint

### OptimizerFactory (services/training/optimizer_training_service.py)
- **Fungsi**: Factory untuk membuat optimizer
- **Metode Utama**:
  - `create(optimizer_type, model_params, **kwargs)`: Buat optimizer
  - `create_optimizer_with_layer_lr(model, base_lr, backbone_lr_factor, optimizer_type, **kwargs)`: Buat optimizer dengan LR berbeda per layer

### SchedulerFactory (services/training/scheduler_training_service.py)
- **Fungsi**: Factory untuk membuat scheduler
- **Metode Utama**:
  - `create(scheduler_type, optimizer, **kwargs)`: Buat scheduler
  - `create_one_cycle_scheduler(optimizer, max_lr, total_steps, pct_start, **kwargs)`: Buat One Cycle LR scheduler

### TrainingCallbacks (services/training/callbacks_training_service.py)
- **Fungsi**: Kelas utilitas untuk callback
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi kumpulan callbacks
  - `add_callback(callback)`: Tambahkan callback ke daftar
  - `execute(metrics)`: Jalankan semua callbacks
  - `create_checkpoint_callback(save_dir, model, prefix, every_n_epochs, save_best_only, monitor, mode, logger)`: Callback untuk checkpoint
  - `create_progress_callback(log_every_n_steps, logger)`: Callback untuk progress
  - `create_tensorboard_callback(log_dir, comment)`: Callback untuk logging ke TensorBoard
  - `create_reduceLR_callback(scheduler, monitor, mode, patience, factor, min_lr, verbose, logger)`: Callback untuk mengurangi learning rate

### EarlyStoppingHandler (services/training/early_stopping_training_service.py)
- **Fungsi**: Handler untuk early stopping
- **Metode Utama**:
  - `__init__(patience, min_delta, monitor, mode, logger)`: Inisialisasi early stopping handler
  - `__call__(metrics)`: Periksa apakah training harus dihentikan
  - `_handle_improvement(current_value)`: Handle kasus ada peningkatan
  - `_handle_no_improvement(current_value)`: Handle kasus tidak ada peningkatan
  - `reset()`: Reset state early stopping

### CosineDecayWithWarmup (services/training/warmup_scheduler_training_service.py)
- **Fungsi**: Cosine learning rate decay dengan warmup phase
- **Metode Utama**:
  - `__init__(optimizer, warmup_epochs, max_epochs, min_lr_factor, last_epoch)`: Inisialisasi scheduler
  - `get_lr()`: Update learning rate berdasarkan schedule

### ExperimentTracker (services/training/experiment_tracker_training_service.py)
- **Fungsi**: Tracking dan visualisasi eksperimen
- **Metode Utama**:
  - `__init__(experiment_name, output_dir, logger)`: Inisialisasi experiment tracker
  - `start_experiment(config)`: Mulai eksperimen baru
  - `log_metrics(epoch, train_loss, val_loss, lr, additional_metrics)`: Catat metrik
  - `end_experiment(final_metrics)`: Akhiri eksperimen
  - `save_metrics()`: Simpan metrik saat ini ke file
  - `load_metrics()`: Muat metrik dari file
  - `plot_metrics(save_to_file)`: Plot metrik training dan validation loss
  - `generate_report()`: Generate laporan eksperimen
  - `list_experiments(output_dir)`: Daftar semua eksperimen yang tersedia
  - `compare_experiments(experiment_names, output_dir, save_to_file)`: Bandingkan beberapa eksperimen

### EvaluationService (services/evaluation/core_evaluation_service.py)
- **Fungsi**: Layanan evaluasi model
- **Metode Utama**:
  - `__init__(config, output_dir, logger, visualizer)`: Inisialisasi layanan evaluasi
  - `evaluate(model, dataloader, conf_thres, iou_thres, max_det, visualize, batch_size, return_samples, experiment_tracker, **kwargs)`: Evaluasi model
  - `evaluate_by_layer(model, dataloader, **kwargs)`: Evaluasi model per layer
  - `evaluate_by_class(model, dataloader, **kwargs)`: Evaluasi model per kelas

### MetricsComputation (services/evaluation/metrics_evaluation_service.py)
- **Fungsi**: Komputasi metrik evaluasi
- **Metode Utama**:
  - `__init__(config, logger)`: Inisialisasi MetricsComputation
  - `reset()`: Reset statistik evaluasi
  - `update(predictions, targets, inference_time)`: Update metrik dengan batch baru
  - `_update_confusion_matrix(layer, predictions, targets, iou_threshold)`: Update confusion matrix
  - `_calculate_batch_metrics()`: Hitung metrik untuk batch terakhir
  - `get_last_batch_metrics()`: Dapatkan metrik dari batch terakhir
  - `compute()`: Hitung metrik evaluasi final
  - `_compute_class_metrics()`: Hitung metrik per kelas
  - `_compute_pr_curves()`: Hitung kurva precision-recall

### PredictionService (services/prediction/core_prediction_service.py)
- **Fungsi**: Layanan prediksi
- **Metode Utama**:
  - `__init__(model, config, logger)`: Inisialisasi service prediksi
  - `predict(images, return_annotated, conf_threshold, iou_threshold)`: Prediksi objek dalam gambar
  - `_preprocess_images(images)`: Preproses gambar untuk inferensi
  - `_postprocess_predictions(predictions, original_images, conf_threshold, iou_threshold)`: Postproses hasil prediksi
  - `predict_from_files(image_paths, return_annotated, conf_threshold, iou_threshold)`: Prediksi dari file gambar
  - `visualize_predictions(image, detections, conf_threshold, output_path)`: Visualisasikan hasil prediksi

### BatchPredictionProcessor (services/prediction/batch_processor_prediction_service.py)
- **Fungsi**: Processor untuk batch prediction
- **Metode Utama**:
  - `__init__(prediction_service, output_dir, num_workers, batch_size, logger)`: Inisialisasi batch prediction processor
  - `process_directory(input_dir, save_results, save_annotated, file_ext, recursive)`: Proses direktori gambar
  - `process_files(files, save_results, save_annotated)`: Proses list file gambar
  - `_save_batch_results(batch_results, batch_idx)`: Simpan hasil batch
  - `run_and_save(input_source, output_filename, save_annotated)`: Jalankan prediksi dan simpan hasil

### ExperimentService (services/experiment/experiment_service.py)
- **Fungsi**: Layanan untuk mengelola eksperimen
- **Metode Utama**:
  - `__init__(experiment_dir, training_service, evaluation_service, logger)`: Inisialisasi experiment service
  - `setup_experiment(name, config, description)`: Setup eksperimen baru
  - `setup_model(model_type, batch_size, learning_rate, **kwargs)`: Setup model untuk eksperimen
  - `run_training(train_loader, val_loader, epochs, callbacks, **kwargs)`: Jalankan training
  - `run_evaluation(test_loader, **kwargs)`: Jalankan evaluasi
  - `run_complete_experiment(train_loader, val_loader, test_loader, model_type, epochs, batch_size, learning_rate, name, **kwargs)`: Jalankan eksperimen lengkap
  - `save_checkpoint(filename)`: Simpan checkpoint model
  - `load_checkpoint(checkpoint_path)`: Muat checkpoint model
  - `save_results(results)`: Simpan hasil eksperimen
  - `load_results()`: Muat hasil eksperimen
  - `predict(inputs, **kwargs)`: Lakukan prediksi dengan model

### ResearchExperimentService (services/research/experiment_service.py)
- **Fungsi**: Facade untuk layanan penelitian model
- **Metode Utama**:
  - `__init__(base_dir, config, logger)`: Inisialisasi experiment service
  - `create_experiment(name, description, config_overrides, tags)`: Buat eksperimen baru
  - `run_experiment(experiment, dataset_path, epochs, batch_size, learning_rate, model_type, callbacks, **kwargs)`: Jalankan eksperimen
  - `run_comparison_experiment(name, dataset_path, models_to_compare, epochs, batch_size, **kwargs)`: Jalankan perbandingan model
  - `run_parameter_tuning(name, dataset_path, model_type, param_grid, **kwargs)`: Jalankan tuning parameter
  - `get_experiment_results(experiment_id)`: Dapatkan hasil eksperimen
  - `list_experiments(filter_tags)`: Dapatkan daftar eksperimen
  - `compare_experiments(experiment_ids, metrics)`: Bandingkan eksperimen
  - `generate_experiment_report(experiment_id, include_plots)`: Generate report eksperimen

### NMSProcessor (services/postprocessing/nms_processor.py)
- **Fungsi**: Processor untuk melakukan Non-Maximum Suppression pada hasil deteksi
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi NMS processor
  - `process(detections, iou_threshold, conf_threshold, class_specific, max_detections)`: Proses deteksi dengan Non-Maximum Suppression


## DOMAIN UI COMPONENTS

### Shared Components

#### Headers (smartcash/components/shared/headers.py)
- **Fungsi**: Komponen header dan section title reusable
- **Metode Utama**:
  - `create_header(title, description, icon)`: Buat komponen header dengan style konsisten
  - `create_section_title(title, icon)`: Buat judul section dengan style konsisten

#### Alerts (smartcash/components/shared/alerts.py)
- **Fungsi**: Komponen alerts, info boxes, dan status indicators
- **Metode Utama**:
  - `create_status_indicator(status, message)`: Buat indikator status dengan style yang sesuai
  - `create_info_alert(message, alert_type, icon)`: Buat alert box dengan style yang sesuai
  - `create_info_box(title, content, style, icon, collapsed)`: Buat info box yang dapat di-collapse

#### Layouts (smartcash/components/shared/layouts.py)
- **Fungsi**: Layout standar untuk widgets UI
- **Konstanta Utama**:
  - `STANDARD_LAYOUTS`: Dictionary layout standar (header, section, container, output, button, hbox, vbox)
  - `MAIN_CONTAINER`: Layout untuk container utama
  - `OUTPUT_WIDGET`: Layout untuk output widget
  - `BUTTON`, `HIDDEN_BUTTON`: Layout untuk tombol
  - `TEXT_INPUT`, `TEXT_AREA`: Layout untuk input text
  - `HORIZONTAL_GROUP`, `VERTICAL_GROUP`: Layout untuk grup widget
  - `CARD`, `TABS`, `ACCORDION`: Layout untuk komponen container
- **Metode Utama**:
  - `create_divider()`: Buat divider horizontal

#### Metrics (smartcash/components/shared/metrics.py)
- **Fungsi**: Komponen UI untuk menampilkan metrik dengan styling konsisten
- **Metode Utama**:
  - `create_metric_display(label, value, unit, is_good)`: Buat display metrik dengan style konsisten
  - `create_result_table(data, title, highlight_max)`: Tampilkan table hasil dengan highlighting
  - `plot_statistics(data, title, kind, figsize, **kwargs)`: Plot statistik data
  - `styled_html(content, bg_color, text_color, border_color, padding, margin)`: Buat HTML dengan styling kustom

#### Helpers (ui_components/shared/helpers.py)
- **Fungsi**: Helper functions untuk komponen UI
- **Metode Utama**:
  - `create_tab_view(tabs)`: Buat komponen Tab dengan konfigurasi otomatis
  - `create_loading_indicator(message)`: Buat indikator loading dengan callback
  - `update_output_area(output_widget, message, status, clear)`: Update area output dengan status baru
  - `register_observer_callback(observer_manager, event_type, output_widget, group_name)`: Register callback for observer events
  - `display_file_info(file_path, description)`: Tampilkan informasi file
  - `create_progress_updater(progress_bar)`: Buat fungsi updater untuk progress bar
  - `run_async_task(task_func, on_complete, on_error, with_output)`: Jalankan task secara asinkron
  - `create_button_group(buttons, layout)`: Buat grup tombol dengan layout konsisten
  - `create_confirmation_dialog(title, message, on_confirm, on_cancel, confirm_label, cancel_label)`: Buat dialog konfirmasi

#### Validators (smartcash/components/shared/validators.py)
- **Fungsi**: Utilitas validasi untuk input UI dan form handling
- **Metode Utama**:
  - `create_validation_message(message, is_error)`: Buat pesan validasi
  - `show_validation_message(container, message, is_error)`: Tampilkan pesan validasi
  - `clear_validation_messages(container)`: Hapus semua pesan validasi
  - `validate_required(value)`: Validasi field tidak boleh kosong
  - `validate_numeric(value)`, `validate_integer(value)`: Validasi nilai numerik/integer
  - `validate_min_value(value, min_value)`, `validate_max_value(value, max_value)`: Validasi nilai min/max
  - `validate_range(value, min_value, max_value)`: Validasi nilai dalam range
  - `validate_min_length(value, min_length)`, `validate_max_length(value, max_length)`: Validasi panjang string
  - `validate_regex(value, pattern, message)`: Validasi string dengan regex
  - `validate_email(value)`, `validate_url(value)`: Validasi format email/URL
  - `validate_file_exists(value)`, `validate_directory_exists(value)`: Validasi file/direktori ada
  - `validate_file_extension(value, allowed_extensions)`: Validasi ekstensi file
  - `validate_api_key(value, min_length)`: Validasi API key
  - `validate_form(form_data, validation_rules)`: Validasi form dengan berbagai aturan
  - `create_validator(validation_func, error_message)`: Buat fungsi validator kustom
  - `combine_validators(*validators)`: Gabungkan beberapa validator

### Dataset Components

#### Download Component (smartcash/components/dataset/download.py)
- **Fungsi**: Komponen UI untuk download dataset
- **Metode Utama**:
  - `create_download_ui(env, config)`: Buat UI untuk download dataset
  - `create_source_selection(env, config)`: Buat komponen pemilihan sumber dataset
  - `create_roboflow_config(env, config)`: Buat konfigurasi Roboflow
  - `create_local_upload(env, config)`: Buat komponen upload dataset lokal

#### Preprocessing Component (ui_components/dataset/preprocessing.py)
- **Fungsi**: Komponen UI untuk preprocessing dataset
- **Metode Utama**:
  - `create_preprocessing_ui(env, config)`: Buat UI untuk preprocessing dataset
  - `create_preprocessing_options(env, config)`: Buat opsi preprocessing
  - `create_preprocessing_preview(env, config)`: Buat komponen preview hasil preprocessing

#### Split Component (smartcash/components/dataset/split.py)
- **Fungsi**: Komponen UI untuk split dataset
- **Metode Utama**:
  - `create_split_ui(env, config)`: Buat UI untuk split dataset
  - `create_split_ratio_selector(env, config)`: Buat komponen pemilihan rasio split
  - `create_stratification_options(env, config)`: Buat opsi stratifikasi

#### Augmentation Component (smartcash/components/dataset/augmentation.py)
- **Fungsi**: Komponen UI untuk augmentasi dataset
- **Metode Utama**:
  - `create_augmentation_ui(env, config)`: Buat UI untuk augmentasi dataset
  - `create_augmentation_options(env, config)`: Buat opsi augmentasi
  - `create_augmentation_preview(env, config)`: Buat komponen preview hasil augmentasi

### Training Config Components

#### Backbone Selection Component (smartcash/components/training_config/backbone_selection.py)
- **Fungsi**: Komponen UI untuk pemilihan backbone model
- **Metode Utama**:
  - `create_backbone_ui(env, config)`: Buat UI untuk pemilihan backbone
  - `create_backbone_selector(env, config)`: Buat komponen pemilihan backbone
  - `create_backbone_info(env, config)`: Buat info tentang backbone yang dipilih

#### Hyperparameters Component (smartcash/components/training_config/hyperparameters.py)
- **Fungsi**: Komponen UI untuk setting hyperparameter
- **Metode Utama**:
  - `create_hyperparameters_ui(env, config)`: Buat UI untuk setting hyperparameter
  - `create_batch_size_selector(env, config)`: Buat komponen pemilihan batch size
  - `create_learning_rate_selector(env, config)`: Buat komponen pemilihan learning rate
  - `create_epochs_selector(env, config)`: Buat komponen pemilihan epochs
  - `create_optimization_options(env, config)`: Buat komponen opsi optimisasi

#### Training Strategy Component (smartcash/components/training_config/training_strategy.py)
- **Fungsi**: Komponen UI untuk strategi training
- **Metode Utama**:
  - `create_training_strategy_ui(env, config)`: Buat UI untuk strategi training
  - `create_scheduler_options(env, config)`: Buat komponen opsi scheduler
  - `create_early_stopping_options(env, config)`: Buat komponen opsi early stopping
  - `create_checkpoint_options(env, config)`: Buat komponen opsi checkpoint

#### Layer Config Component (smartcash/components/training_config/layer_config.py)
- **Fungsi**: Komponen UI untuk konfigurasi layer deteksi
- **Metode Utama**:
  - `create_layer_config_ui(env, config)`: Buat UI untuk konfigurasi layer
  - `create_layer_selector(env, config)`: Buat komponen pemilihan layer
  - `create_class_config(env, config)`: Buat komponen konfigurasi class

### Training Execution Components

#### Model Training Component (smartcash/components/training_execution/model_training.py)
- **Fungsi**: Komponen UI untuk pelatihan model
- **Metode Utama**:
  - `create_model_training_ui(env, config)`: Buat UI untuk pelatihan model
  - `create_training_controls(env, config)`: Buat komponen kontrol training
  - `create_training_progress(env, config)`: Buat komponen progress training
  - `create_training_output(env, config)`: Buat komponen output training

#### Performance Tracking Component (smartcash/components/training_execution/performance_tracking.py)
- **Fungsi**: Komponen UI untuk tracking performa
- **Metode Utama**:
  - `create_performance_tracking_ui(env, config)`: Buat UI untuk tracking performa
  - `create_metrics_display(env, config)`: Buat komponen display metrik
  - `create_loss_chart(env, config)`: Buat komponen chart loss
  - `create_metrics_history(env, config)`: Buat komponen history metrik

#### Checkpoint Management Component (smartcash/components/training_execution/checkpoint_management.py)
- **Fungsi**: Komponen UI untuk manajemen checkpoint
- **Metode Utama**:
  - `create_checkpoint_management_ui(env, config)`: Buat UI untuk manajemen checkpoint
  - `create_checkpoint_list(env, config)`: Buat komponen daftar checkpoint
  - `create_checkpoint_actions(env, config)`: Buat komponen aksi checkpoint
  - `create_checkpoint_details(env, config)`: Buat komponen detail checkpoint

### Model Evaluation Components

#### Performance Metrics Component (smartcash/components/model_evaluation/performance_metrics.py)
- **Fungsi**: Komponen UI untuk metrik performa
- **Metode Utama**:
  - `create_performance_metrics_ui(env, config)`: Buat UI untuk metrik performa
  - `create_metrics_dashboard(env, config)`: Buat dashboard metrik
  - `create_confusion_matrix(env, config)`: Buat komponen confusion matrix
  - `create_precision_recall_curve(env, config)`: Buat komponen kurva precision-recall

#### Comparative Analysis Component (smartcash/components/model_evaluation/comparative_analysis.py)
- **Fungsi**: Komponen UI untuk analisis komparatif
- **Metode Utama**:
  - `create_comparative_analysis_ui(env, config)`: Buat UI untuk analisis komparatif
  - `create_model_selector(env, config)`: Buat komponen pemilihan model
  - `create_comparison_chart(env, config)`: Buat komponen chart perbandingan
  - `create_comparison_table(env, config)`: Buat komponen tabel perbandingan

#### Visualization Component (smartcash/components/model_evaluation/visualization.py)
- **Fungsi**: Komponen UI untuk visualisasi hasil
- **Metode Utama**:
  - `create_visualization_ui(env, config)`: Buat UI untuk visualisasi hasil
  - `create_detection_visualizer(env, config)`: Buat komponen visualisasi deteksi
  - `create_layer_visualizer(env, config)`: Buat komponen visualisasi layer
  - `create_export_options(env, config)`: Buat komponen opsi export

## DOMAIN UI HANDLERS