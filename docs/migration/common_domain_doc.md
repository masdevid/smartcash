# SmartCash v2 - Domain Common

## 1. File Structure
```
smartcash/common/
├── init.py                # Ekspor utilitas umum dengan all
├── visualization/             # Visualisasi
│   ├── init.py            # Ekspor komponen visualisasi
│   ├── core/                  # Core visualisasi
│   │   ├── init.py        # Ekspor komponen core
│   │   └── visualization_base.py # Base class untuk visualisasi
│   └── helpers/               # Komponen visualisasi
│       ├── init.py        # Ekspor komponen helper visualisasi
│       ├── chart_helper.py    # ChartHelper: Visualisasi chart
│       ├── color_helper.py    # ColorHelper: Visualisasi warna
│       ├── annotation_helper.py # AnnotationHelper: Visualisasi anotasi
│       ├── export_helper.py   # ExportHelper: Export visualisasi
│       ├── layout_helper.py   # LayoutHelper: Layout visualisasi
│       └── style_helper.py    # StyleHelper: Styling visualisasi
├── interfaces/                # Abstract interfaces
│   ├── init.py            # Ekspor interfaces
│   ├── visualization_interface.py # Interface untuk visualisasi
│   ├── layer_config_interface.py  # Interface untuk konfigurasi layer
│   └── checkpoint_interface.py    # Interface untuk checkpoint
├── config.py                  # ConfigManager: Manager konfigurasi multi-format
├── constants.py               # Konstanta global (VERSION, APP_NAME, dll)
├── logger.py                  # SmartCashLogger: Logger dengan emojis, warna, callback
├── exceptions.py              # Exception hierarchy
├── types.py                   # Type definitions
├── utils.py                   # Fungsi utilitas umum
├── layer_config.py            # LayerConfigManager: Konfigurasi layer deteksi
└── environment.py            # EnvironmentManager: Manajer lingkungan
```

## 2. Class and Methods Mapping

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

### Exceptions (smartcash/common/exceptions.py)
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