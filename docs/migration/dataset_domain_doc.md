# SmartCash v2 - Domain Dataset

## 1. File Structure
```
smartcash/dataset/
├── init.py                 # Ekspor komponen dataset
├── manager.py                  # DatasetManager: Koordinator alur kerja dataset
├── services/
│   ├── init.py
│   ├── loader/                 # Layanan loading dataset
│   │   ├── init.py
│   │   ├── dataset_loader.py   # DatasetLoader: Loading dataset dari disk
│   │   ├── multilayer_loader.py # MultilayerLoader: Loader untuk dataset multilayer
│   │   ├── cache_manager.py    # DatasetCacheManager: Cache untuk dataset
│   │   └── batch_generator.py  # BatchGenerator: Generator batch data
│   │   ├── preprocessed_dataset_loader.py # PreprocessedDatasetLoader: Loading dataset hasil preprocessing
│   ├── validator/              # Layanan validasi dataset
│   │   ├── init.py
│   │   ├── dataset_validator.py # DatasetValidator: Validasi dataset utama
│   │   ├── label_validator.py  # LabelValidator: Validasi file label
│   │   ├── image_validator.py  # ImageValidator: Validasi gambar
│   │   └── fixer.py            # DatasetFixer: Perbaikan dataset
│   ├── preprocessor/           # Layanan preprocessing dataset
│   │   ├── init.py
│   │   ├── dataset_preprocessor.py   # Koordinator utama preprocessing dataset
│   │   ├── pipeline.py               # Pipeline transformasi preprocessing
│   │   ├── storage.py                # Pengelolaan file hasil preprocessing
│   │   └── cleaner.py                # Pembersih cache preprocessed
│   ├── augmentor/              # Layanan augmentasi dataset
│   │   ├── init.py
│   │   ├── augmentation_service.py # AugmentationService: Layanan augmentasi
│   │   ├── image_augmentor.py  # ImageAugmentor: Augmentasi gambar
│   │   ├── bbox_augmentor.py   # BBoxAugmentor: Augmentasi bounding box
│   │   └── pipeline_factory.py # AugmentationPipelineFactory: Factory pipeline
│   ├── downloader/             # Layanan download dataset
│   │   ├── init.py
│   │   ├── download_service.py # DownloadService: Service utama download
│   │   ├── roboflow_downloader.py # RoboflowDownloader: Download dari Roboflow
│   │   ├── download_validator.py # DownloadValidator: Validasi integritas download
│   │   └── file_processor.py   # FileProcessor: Pemrosesan file dataset
│   ├── explorer/               # Layanan eksplorasi dataset
│   │   ├── init.py
│   │   ├── explorer_service.py # ExplorerService: Layanan eksplorasi
│   │   ├── class_explorer.py   # ClassExplorer: Eksplorasi distribusi kelas
│   │   ├── layer_explorer.py   # LayerExplorer: Eksplorasi distribusi layer
│   │   ├── bbox_explorer.py    # BBoxExplorer: Eksplorasi bounding box
│   │   └── image_explorer.py   # ImageExplorer: Eksplorasi gambar
│   ├── balancer/               # Layanan balancing dataset
│   │   ├── init.py
│   │   ├── balance_service.py  # BalanceService: Layanan balancing
│   │   ├── undersampler.py     # Undersampler: Undersampling dataset
│   │   ├── oversampler.py      # Oversampler: Oversampling dataset
│   │   └── weight_calculator.py # WeightCalculator: Perhitungan bobot kelas
│   └── reporter/               # Layanan pelaporan dataset
│       ├── init.py
│       ├── report_service.py   # ReportService: Layanan pelaporan dataset
│       ├── metrics_reporter.py # MetricsReporter: Pelaporan metrik
│       ├── export_formatter.py # ExportFormatter: Format ekspor laporan
│       └── visualization_service.py # VisualizationService: Visualisasi metrik dan laporan
├── utils/
│   ├── init.py
│   ├── transform/              # Transformasi dataset
│   │   ├── init.py
│   │   ├── albumentations_adapter.py # AlbumentationsAdapter: Adapter Albumentations
│   │   ├── bbox_transform.py   # BBoxTransformer: Transformasi bounding box
│   │   ├── image_transform.py  # ImageTransformer: Transformasi gambar
│   │   ├── polygon_transform.py # PolygonTransformer: Transformasi polygon
│   │   └── format_converter.py # FormatConverter: Konversi format
│   ├── split/                  # Utilitas split dataset
│   │   ├── init.py
│   │   ├── dataset_splitter.py # DatasetSplitter: Split dataset
│   │   ├── merger.py           # DatasetMerger: Merge dataset
│   │   └── stratifier.py       # DatasetStratifier: Stratified split
│   ├── statistics/             # Statistik dataset
│   │   ├── init.py
│   │   ├── class_stats.py      # ClassStatistics: Statistik kelas
│   │   ├── image_stats.py      # ImageStatistics: Statistik gambar
│   │   └── distribution_analyzer.py # DistributionAnalyzer: Analisis distribusi
│   ├── file/                   # Pemrosesan file
│   │   ├── init.py
│   │   ├── file_processor.py   # FileProcessor: Processor file umum
│   │   ├── image_processor.py  # ImageProcessor: Processor gambar
│   │   └── label_processor.py  # LabelProcessor: Processor label
│   └── progress/               # Tracking progres
│       ├── init.py
│       ├── progress_tracker.py # ProgressTracker: Tracking progres
│       └── observer_adapter.py # ProgressObserver: Observer untuk progres
└── components/
├── init.py
├── datasets/               # Komponen dataset
│   ├── init.py
│   ├── base_dataset.py     # BaseDataset: Dataset dasar
│   ├── multilayer_dataset.py # MultilayerDataset: Dataset multilayer
│   └── yolo_dataset.py     # YOLODataset: Dataset format YOLO
├── geometry/               # Komponen geometri
│   ├── init.py
│   ├── polygon_handler.py  # PolygonHandler: Handler polygon
│   ├── coord_converter.py  # CoordinateConverter: Konversi koordinat
│   └── geometry_utils.py   # Utilitas geometri (IoU, area, clip)
├── labels/                 # Komponen label
│   ├── init.py
│   ├── label_handler.py    # LabelHandler: Handler label
│   ├── multilayer_handler.py # MultilayerLabelHandler: Handler label multilayer
│   └── format_converter.py # LabelFormatConverter: Konversi format label
├── samplers/               # Komponen sampler
│   ├── init.py
│   ├── balanced_sampler.py # BalancedBatchSampler: Sampler dengan balance kelas
│   └── weighted_sampler.py # WeightedRandomSampler: Sampler dengan bobot
└── collate/                # Komponen collate function
├── init.py
├── multilayer_collate.py # multilayer_collate_fn(): Collate untuk multilayer
└── yolo_collate.py     # yolo_collate_fn(): Collate untuk YOLO
```

## 2. Class and Methods Mapping

### DatasetManager (manager.py)
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