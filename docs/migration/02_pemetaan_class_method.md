# SmartCash v2 - Pemetaan Kelas dan Fungsi yang Diperbarui

## Domain Common (Tidak Berubah)

### ConfigManager (config.py)
- **Fungsi**: Mengelola konfigurasi aplikasi
- **Metode Utama**:
  - `load_config(config_file)`: Memuat konfigurasi dari file YAML/JSON
  - `get(key, default)`: Mengambil nilai dengan dot notation
  - `set(key, value)`: Mengatur nilai dengan dot notation
  - `merge_config(config)`: Menggabungkan konfigurasi dari dict/file
  - `save_config(config_file)`: Menyimpan konfigurasi ke file

### SmartCashLogger (logger.py)
- **Fungsi**: Sistem logging dengan emoji dan warna
- **Metode Utama**:
  - `log(level, message)`: Mencatat pesan log
  - `add_callback(callback)`: Tambah callback untuk event log
  - `debug/info/success/warning/error/critical()`: Shortcut log level
  - `progress(iterable)`: Membuat progress bar dengan tqdm

### LayerConfigManager (layer_config.py)
- **Fungsi**: Mengelola konfigurasi layer deteksi
- **Metode Utama**:
  - `get_layer_config(layer_name)`: Config untuk layer tertentu
  - `get_class_map()`: Mapping class_id ke class_name
  - `get_layer_for_class_id(class_id)`: Layer untuk class_id
  - `update_layer_config(layer_name, config)`: Update config layer
  - `validate_class_ids()`: Validasi class_id (duplikat, gap)

### ChartHelper (common/visualization/helpers/chart_helper.py)
- **Fungsi**: Visualisasi berbasis chart
- **Metode Utama**:
  - `create_bar_chart(data, title, xlabel, ylabel)`: Buat bar chart
  - `create_line_chart(data, title, xlabel, ylabel)`: Buat line chart
  - `create_pie_chart(data, title, labels)`: Buat pie chart
  - `create_scatter_plot(x, y, title, xlabel, ylabel)`: Buat scatter plot
  - `create_heatmap(data, title, xlabel, ylabel)`: Buat heatmap

### ColorHelper (common/visualization/helpers/color_helper.py)
- **Fungsi**: Manajemen warna untuk visualisasi
- **Metode Utama**:
  - `get_color_palette(palette_name, n_colors)`: Dapatkan palette warna
  - `get_categorical_colors(n_categories)`: Warna untuk kategori
  - `get_sequential_colors(n_colors, start_color, end_color)`: Warna berurutan
  - `get_color_for_value(value, min_val, max_val)`: Warna berdasarkan nilai
  - `get_class_colors(class_names)`: Warna konsisten untuk kelas

### AnnotationHelper (common/visualization/helpers/annotation_helper.py)
- **Fungsi**: Anotasi pada visualisasi
- **Metode Utama**:
  - `add_text_annotations(ax, data, format_str)`: Tambah anotasi teks
  - `add_arrow_annotations(ax, points, texts)`: Tambah anotasi panah
  - `add_bbox_annotations(ax, boxes, labels)`: Tambah anotasi bbox
  - `add_value_labels(ax, spacing, format_str)`: Tambah label nilai
  - `add_statistical_annotations(ax, data, test, loc)`: Tambah anotasi statistik

### ExportHelper (common/visualization/helpers/export_helper.py)
- **Fungsi**: Export visualisasi
- **Metode Utama**:
  - `save_figure(fig, output_path, dpi)`: Simpan figure
  - `save_interactive_plot(fig, output_path)`: Simpan plot interaktif
  - `export_to_html(fig, output_path)`: Export ke HTML
  - `export_to_notebook(fig)`: Export ke notebook
  - `create_report_from_figures(figures, output_path)`: Buat laporan

### LayoutHelper (common/visualization/helpers/layout_helper.py)
- **Fungsi**: Layout untuk visualisasi
- **Metode Utama**:
  - `create_grid_layout(nrows, ncols, figsize)`: Buat layout grid
  - `create_dashboard_layout(areas, figsize)`: Buat layout dashboard
  - `adjust_subplots(fig, wspace, hspace)`: Atur spacing subplot
  - `add_suptitle(fig, title, fontsize)`: Tambah judul utama
  - `create_nested_layout(fig, areas)`: Buat layout bersarang

### StyleHelper (common/visualization/helpers/style_helper.py)
- **Fungsi**: Styling visualisasi
- **Metode Utama**:
  - `set_plot_style(style_name)`: Set style plot
  - `apply_custom_theme(fig, theme_name)`: Terapkan tema kustom
  - `set_fonts(font_family, title_size, label_size)`: Set font
  - `set_grid_style(ax, grid_style)`: Set style grid
  - `set_legend_style(ax, loc, frameon)`: Set style legend
## Domain Components (Tidak Berubah)

### EventDispatcher (event_dispatcher_observer.py)
- **Fungsi**: Mengelola notifikasi event ke observer
- **Metode Utama**:
  - `register(event_type, observer)`: Daftarkan observer
  - `notify(event_type, sender, **kwargs)`: Kirim notifikasi
  - `unregister_from_all(observer)`: Batalkan registrasi
  - `get_stats()`: Dapatkan statistik dispatcher

### BaseObserver (base_observer.py)
- **Fungsi**: Kelas dasar untuk implementasi observer
- **Metode Utama**:
  - `update(event_type, sender, **kwargs)`: Handler event (abstract)
  - `should_process_event(event_type)`: Cek filter event
  - `enable()/disable()`: Aktifkan/nonaktifkan observer

### ObserverManager (manager_observer.py)
- **Fungsi**: Mengelola observer dengan sistem grup
- **Metode Utama**:
  - `create_simple_observer(event_type, callback)`: Buat observer
  - `create_progress_observer(event_types, total)`: Observer progress
  - `unregister_group(group)`: Batalkan registrasi grup
  - `get_observers_by_group(group)`: Observer dalam grup

### CacheManager (manager_cache.py)
- **Fungsi**: Mengelola sistem caching terpusat
- **Metode Utama**:
  - `get(key)`: Ambil data dari cache
  - `put(key, data)`: Simpan data ke cache
  - `cleanup(expired_only)`: Bersihkan cache
  - `get_stats()`: Dapatkan statistik cache

## Domain Dataset (Diperbarui)

### DatasetManager (manager.py)
- **Fungsi**: Koordinator alur kerja dataset
- **Metode Utama**:
  - `get_service(service_name)`: Lazy-initialization service dataset
  - `get_dataset(split, **kwargs)`: Dapatkan dataset untuk split tertentu
  - `get_dataloader(split, **kwargs)`: Dapatkan dataloader untuk split
  - `get_all_dataloaders(**kwargs)`: Dapatkan semua dataloader
  - `validate_dataset(split, **kwargs)`: Validasi dataset
  - `fix_dataset(split, **kwargs)`: Perbaiki masalah dataset
  - `augment_dataset(**kwargs)`: Augmentasi dataset
  - `download_from_roboflow(**kwargs)`: Download dataset dari Roboflow
  - `upload_local_dataset(zip_path, **kwargs)`: Upload dataset lokal
  - `explore_class_distribution(split)`: Analisis distribusi kelas
  - `explore_layer_distribution(split)`: Analisis distribusi layer
  - `explore_bbox_statistics(split)`: Analisis statistik bbox
  - `balance_dataset(split, **kwargs)`: Seimbangkan dataset
  - `generate_dataset_report(splits, **kwargs)`: Buat laporan dataset
  - `visualize_class_distribution(class_stats, **kwargs)`: Visualisasi distribusi kelas
  - `visualize_sample_images(data_dir, **kwargs)`: Visualisasi sampel gambar
  - `create_dataset_dashboard(report, **kwargs)`: Buat dashboard visualisasi
  - `get_split_statistics()`: Dapatkan statistik dasar split
  - `split_dataset(**kwargs)`: Pecah dataset menjadi train/valid/test

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

## Dataset Components

### MultilayerDataset (components/datasets/multilayer_dataset.py)
- **Fungsi**: Dataset multilayer untuk deteksi
- **Metode Utama**:
  - `__getitem__(idx)`: Ambil item dataset
  - `get_layer_annotation(img_id, layer)`: Anotasi per layer

### DatasetUtils (utils/dataset_utils.py)
- **Fungsi**: Utilitas untuk operasi dataset
- **Metode Utama**:
  - `get_split_path(split)`: Dapatkan path untuk split dataset
  - `get_class_name(cls_id)`: Dapatkan nama kelas dari ID
  - `get_layer_from_class(cls_id)`: Dapatkan layer dari class ID
  - `find_image_files(directory, with_labels)`: Cari file gambar
  - `get_random_sample(items, sample_size, seed)`: Ambil sampel acak
  - `load_image(image_path, target_size)`: Baca gambar dari file
  - `parse_yolo_label(label_path)`: Parse file label YOLO
  - `get_available_layers(label_path)`: Dapatkan layer dalam label
  - `get_split_statistics(splits)`: Dapatkan statistik split
  - `backup_directory(source_dir, suffix)`: Buat backup direktori
  - `move_invalid_files(source_dir, target_dir, file_list)`: Pindahkan file invalid

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
- **Fungsi**: Memecah dataset menjadi train/valid/test
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

## Dataset Utils - Statistik dan Laporan

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
  - `_analyze_images(image_files)`: Analisis ukuran gambar
  - `_analyze_image_quality(image_files)`: Analisis kualitas gambar

### DistributionAnalyzer (utils/statistics/distribution_analyzer.py)
- **Fungsi**: Analisis distribusi statistik dataset
- **Metode Utama**:
  - `analyze_dataset(splits, sample_size)`: Analisis komprehensif dataset
  - `_analyze_cross_split_consistency(results)`: Analisis konsistensi antar split
  - `_generate_suggestions(results)`: Buat rekomendasi dari hasil analisis
  - `_log_summary(results)`: Log ringkasan hasil analisis

### FileProcessor (utils/file/file_processor.py)
- **Fungsi**: Pemrosesan file dan direktori
- **Metode Utama**:
  - `count_files(directory, extensions)`: Hitung jumlah file
  - `copy_files(source_dir, target_dir, file_list)`: Salin file
  - `move_files(source_dir, target_dir, file_list)`: Pindahkan file
  - `extract_zip(zip_path, output_dir, include_patterns)`: Ekstrak zip
  - `merge_splits(source_dir, target_dir, splits)`: Gabung split
  - `find_corrupted_images(directory, recursive)`: Temukan gambar rusak

### ImageProcessor (utils/file/image_processor.py)
- **Fungsi**: Pemrosesan dan manipulasi gambar
- **Metode Utama**:
  - `resize_images(directory, target_size, output_dir)`: Resize gambar
  - `enhance_images(directory, output_dir, enhance_contrast)`: Tingkatkan kualitas
  - `convert_format(directory, target_format, output_dir)`: Konversi format
  - `create_thumbnails(directory, thumbnail_size, output_dir)`: Buat thumbnail

### LabelProcessor (utils/file/label_processor.py)
- **Fungsi**: Pemrosesan dan manipulasi file label
- **Metode Utama**:
  - `fix_labels(directory, fix_coordinates, fix_class_ids)`: Perbaiki label
  - `filter_classes(directory, keep_classes, remove_classes)`: Filter kelas
  - `extract_layer(directory, layer_name, output_dir)`: Ekstrak satu layer
  - `convert_to_coco(dataset_dir, output_file, split)`: Konversi ke COCO

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

## Domain Detection (Diperbarui)

### Detector (detector.py)
- **Fungsi**: Koordinator utama proses deteksi
- **Metode Utama**:
  - `detect(image, **kwargs)`: Deteksi pada gambar
  - `detect_multilayer(image, threshold)`: Deteksi multilayer
  - `detect_batch(images)`: Deteksi pada batch gambar
  - `visualize(image, detections)`: Visualisasi hasil dengan visualisasi terpadu dari domain model

### DetectionVisualizationAdapter (services/visualization_adapter.py)
- **Fungsi**: Adapter untuk mengintegrasikan visualisasi dari domain model ke domain detection
- **Metode Utama**:
  - `visualize_detection(image, detections)`: Visualisasi deteksi menggunakan DetectionVisualizer
  - `visualize_confusion_matrix(cm, class_names)`: Visualisasi confusion matrix menggunakan MetricsVisualizer
  - `visualize_model_comparison(comparison_data)`: Visualisasi perbandingan model menggunakan MetricsVisualizer

### InferenceService (services/inference/inference_service.py)
- **Fungsi**: Mengelola proses inferensi model
- **Metode Utama**:
  - `infer(image)`: Inferensi pada gambar
  - `batch_infer(images)`: Inferensi pada batch
  - `_optimize_model()`: Optimasi model untuk inferensi

### NMSProcessor (services/postprocessing/nms_processor.py)
- **Fungsi**: Pemrosesan Non-Maximum Suppression
- **Metode Utama**:
  - `process(detections, iou_threshold)`: Proses NMS

### DetectionEvaluator (services/evaluation/evaluator.py)
- **Fungsi**: Evaluasi model deteksi
- **Metode Utama**:
  - `evaluate(iou_threshold, conf_threshold)`: Evaluasi model
  - `generate_report()`: Laporan evaluasi

### SmartCashYOLOv5 (models/yolov5_model.py)
- **Fungsi**: Model YOLOv5 dengan EfficientNet backbone
- **Metode Utama**:
  - `forward(x)`: Forward pass model
  - `predict(x, conf_threshold, nms_threshold)`: Prediksi dengan post-processing
  - `get_optimizer(learning_rate, weight_decay)`: Buat optimizer
  - `compute_loss(predictions, targets, loss_fn)`: Hitung loss

## Domain Model (Diperbarui)

### ModelManager (manager.py)
- **Fungsi**: Koordinator alur kerja model
- **Metode Utama**:
  - `create_model(model_type, **kwargs)`: Factory untuk membuat model
  - `build_model()`: Buat dan inisialisasi model
  - `update_config(config_updates)`: Update konfigurasi model
  - `get_config()`: Dapatkan konfigurasi model

### ModelCheckpointManager (manager_checkpoint.py)
- **Fungsi**: Integrasi checkpoint service dengan model manager
- **Metode Utama**:
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint
  - `load_checkpoint(path, model, optimizer, map_location)`: Load checkpoint
  - `export_to_onnx(output_path, input_shape, opset_version)`: Export ke ONNX

### VisualizationHelper (visualization/base_visualizer.py)
- **Fungsi**: Utilitas dasar untuk visualisasi
- **Metode Utama**:
  - `set_plot_style(style)`: Set style untuk matplotlib plots
  - `save_figure(fig, filepath, dpi)`: Simpan figure matplotlib
  - `create_output_directory(output_dir)`: Buat direktori output

### DetectionVisualizer (visualization/detection_visualizer.py)
- **Fungsi**: Visualisasi hasil deteksi objek
- **Metode Utama**:
  - `visualize_detection(image, detections, filename, conf_threshold)`: Visualisasi deteksi pada gambar
  - `visualize_detections_grid(images, detections_list, title, filename)`: Visualisasi batch deteksi dalam grid
  - `calculate_denomination_total(detections)`: Hitung total nilai mata uang dari deteksi

### MetricsVisualizer (visualization/metrics_visualizer.py)
- **Fungsi**: Visualisasi metrik evaluasi model
- **Metode Utama**:
  - `plot_confusion_matrix(cm, class_names, title, filename)`: Plot confusion matrix
  - `plot_training_metrics(metrics, title, filename)`: Plot metrik training
  - `plot_model_comparison(comparison_data, metric_cols, title)`: Plot perbandingan metrik model
  - `plot_research_comparison(results_df, metric_cols, title)`: Plot perbandingan hasil skenario penelitian

### EvaluationVisualizer (visualization/evaluation_visualizer.py)
- **Fungsi**: Visualisasi hasil evaluasi model
- **Metode Utama**:
  - `create_all_plots(metrics_data, prefix)`: Buat semua visualisasi yang tersedia
  - `plot_map_f1_comparison(metrics_df, prefix)`: Plot perbandingan mAP dan F1
  - `plot_inference_time(metrics_df, prefix)`: Plot perbandingan waktu inferensi
  - `plot_backbone_comparison(metrics_df, prefix)`: Plot perbandingan backbone
  - `plot_condition_comparison(metrics_df, prefix)`: Plot perbandingan kondisi pengujian
  - `plot_combined_heatmap(metrics_df, prefix)`: Plot heatmap kombinasi

### ExperimentVisualizer (visualization/experiment_visualizer.py)
- **Fungsi**: Visualisasi hasil eksperimen model
- **Metode Utama**:
  - `visualize_backbone_comparison(results, metrics, title)`: Visualisasi perbandingan antar backbone
  - `visualize_training_curves(metrics_history, title)`: Visualisasi kurva training dan validasi
  - `visualize_parameter_comparison(results, parameter_name, metrics)`: Visualisasi perbandingan hasil dengan parameter berbeda

### ScenarioVisualizer (visualization/scenario_visualizer.py)
- **Fungsi**: Visualisasi dan analisis hasil skenario penelitian
- **Metode Utama**:
  - `visualize_scenario_comparison(results_df, title, filename)`: Visualisasi perbandingan berbagai skenario penelitian
  - `_create_accuracy_plot(ax, df, metric_cols, backbone_col)`: Buat plot akurasi per skenario
  - `_create_inference_time_plot(ax, df, time_col, backbone_col)`: Buat plot waktu inferensi

### ResearchVisualizer (visualization/research_visualizer.py)
- **Fungsi**: Visualisasi dan analisis hasil penelitian
- **Metode Utama**:
  - `visualize_experiment_comparison(results_df, title, filename)`: Visualisasi perbandingan berbagai eksperimen
  - `visualize_scenario_comparison(results_df, title, filename)`: Visualisasi perbandingan berbagai skenario penelitian

### CheckpointService (services/checkpoint/checkpoint_service.py)
- **Fungsi**: Layanan untuk mengelola checkpoint model
- **Metode Utama**:
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint
  - `load_checkpoint(path, model, optimizer, map_location)`: Load checkpoint
  - `list_checkpoints(sort_by)`: Daftar checkpoint yang tersedia
  - `get_best_checkpoint()`: Dapatkan checkpoint terbaik

### TrainingService (services/training/core_training_service.py)
- **Fungsi**: Layanan pelatihan model
- **Metode Utama**:
  - `train(train_loader, val_loader, epochs, callbacks)`: Training model
  - `_train_epoch(train_loader)`: Proses training untuk satu epoch
  - `_validate_epoch(val_loader)`: Proses validasi untuk satu epoch
  - `resume_from_checkpoint(checkpoint_path)`: Lanjutkan training dari checkpoint

### OptimizerFactory (services/training/optimizer_training_service.py)
- **Fungsi**: Factory untuk membuat optimizer
- **Metode Utama**:
  - `create(optimizer_type, model_params, **kwargs)`: Buat optimizer
  - `create_optimizer_with_layer_lr(model, base_lr, backbone_lr_factor)`: Optimizer dengan LR berbeda

### SchedulerFactory (services/training/scheduler_training_service.py)
- **Fungsi**: Factory untuk membuat scheduler
- **Metode Utama**:
  - `create(scheduler_type, optimizer, **kwargs)`: Buat scheduler
  - `create_one_cycle_scheduler(optimizer, max_lr, total_steps)`: One Cycle LR scheduler

### EarlyStoppingHandler (services/training/early_stopping_training_service.py)
- **Fungsi**: Handler untuk early stopping
- **Metode Utama**:
  - `__call__(metrics)`: Cek apakah training harus dihentikan
  - `_handle_improvement(current_value)`: Handle kasus ada peningkatan
  - `_handle_no_improvement(current_value)`: Handle kasus tidak ada peningkatan

### TrainingCallbacks (services/training/callbacks_training_service.py)
- **Fungsi**: Kelas utilitas untuk mengelola callback
- **Metode Utama**:
  - `add_callback(callback)`: Tambahkan callback ke daftar
  - `execute(metrics)`: Jalankan semua callbacks
  - `create_checkpoint_callback(save_dir, model, prefix)`: Callback untuk checkpoint
  - `create_progress_callback(log_every_n_steps)`: Callback untuk progress

### CosineDecayWithWarmup (services/training/warmup_scheduler_training_service.py)
- **Fungsi**: Scheduler dengan fase warmup
- **Metode Utama**:
  - `get_lr()`: Update learning rate berdasarkan schedule

### ExperimentTracker (services/training/experiment_tracker_training_service.py)
- **Fungsi**: Tracking dan visualisasi eksperimen
- **Metode Utama**:
  - `start_experiment(config)`: Mulai eksperimen baru
  - `log_metrics(epoch, train_loss, val_loss, lr)`: Catat metrik
  - `end_experiment(final_metrics)`: Akhiri eksperimen
  - `generate_report()`: Generate laporan eksperimen

### EvaluationService (services/evaluation/core_evaluation_service.py)
- **Fungsi**: Layanan evaluasi model
- **Metode Utama**:
  - `evaluate(model, dataloader, conf_thres, iou_thres)`: Evaluasi model
  - `evaluate_by_layer(model, dataloader)`: Evaluasi model per layer
  - `evaluate_by_class(model, dataloader)`: Evaluasi model per kelas

### MetricsComputation (services/evaluation/metrics_evaluation_service.py)
- **Fungsi**: Komputasi metrik evaluasi
- **Metode Utama**:
  - `update(predictions, targets, inference_time)`: Update metrik dengan batch baru
  - `compute()`: Hitung metrik evaluasi final
  - `get_confusion_matrix(normalized)`: Dapatkan confusion matrix

### PredictionService (services/prediction/core_prediction_service.py)
- **Fungsi**: Layanan prediksi untuk model
- **Metode Utama**:
  - `predict(images, return_annotated)`: Buat prediksi untuk gambar
  - `predict_from_files(image_paths, return_annotated)`: Prediksi dari file
  - `_preprocess_images(images)`: Preproses gambar untuk inferensi
  - `_postprocess_predictions(predictions, original_images)`: Postproses hasil prediksi

### BatchPredictionProcessor (services/prediction/batch_processor_prediction_service.py)
- **Fungsi**: Processor untuk batch prediksi
- **Metode Utama**:
  - `process_directory(input_dir, save_results, save_annotated)`: Proses semua gambar dalam direktori
  - `process_files(files, save_results, save_annotated)`: Proses list file gambar
  - `run_and_save(input_source, output_filename, save_annotated)`: Jalankan prediksi dan simpan hasil

### ExperimentService (services/experiment/experiment_service.py)
- **Fungsi**: Layanan untuk mengelola eksperimen
- **Metode Utama**:
  - `setup_experiment(name, config, description)`: Setup eksperimen baru
  - `setup_model(model_type, batch_size, learning_rate)`: Setup model untuk eksperimen
  - `run_training(train_loader, val_loader, epochs, callbacks)`: Jalankan training
  - `run_evaluation(test_loader)`: Jalankan evaluasi
  - `run_complete_experiment(train_loader, val_loader, test_loader)`: Jalankan eksperimen lengkap

### ExperimentCreator (services/research/experiment_creator.py)
- **Fungsi**: Membuat dan mengelola konfigurasi eksperimen
- **Metode Utama**:
  - `create_experiment(name, description, config_overrides, tags)`: Buat eksperimen baru
  - `create_experiment_group(name, group_type)`: Buat grup eksperimen

### ExperimentRunner (services/research/experiment_runner.py)
- **Fungsi**: Menjalankan eksperimen model
- **Metode Utama**:
  - `run_experiment(experiment, dataset_path, epochs, batch_size, learning_rate)`: Jalankan eksperimen
  - `_setup_model(model_type, batch_size, learning_rate)`: Setup model untuk eksperimen
  - `_execute_training(model, train_loader, val_loader, epochs)`: Jalankan proses training
  - `_execute_evaluation(model, test_loader)`: Jalankan evaluasi

### ExperimentAnalyzer (services/research/experiment_analyzer.py)
- **Fungsi**: Menganalisis hasil eksperimen
- **Metode Utama**:
  - `analyze_experiment_results(df, metric_cols, time_col)`: Analisis hasil eksperimen
  - `_identify_best_model(best_row, metric, model_col, idx)`: Identifikasi model terbaik
  - `_generate_recommendation(analysis)`: Buat rekomendasi berdasarkan analisis

### ParameterTuner (services/research/parameter_tuner.py)
- **Fungsi**: Melakukan tuning parameter model
- **Metode Utama**:
  - `run_parameter_tuning(name, dataset_path, model_type, param_grid)`: Jalankan tuning parameter
  - `_generate_param_combinations(param_grid)`: Generate kombinasi parameter dari grid
  - `_get_best_params(tuning_df)`: Dapatkan parameter terbaik

### ComparisonRunner (services/research/comparison_runner.py)
- **Fungsi**: Menjalankan eksperimen perbandingan model
- **Metode Utama**:
  - `run_comparison_experiment(name, dataset_path, models_to_compare)`: Jalankan eksperimen perbandingan
  - `_run_model_comparison_experiments(name, dataset_path, models_to_compare)`: Jalankan eksperimen untuk setiap model
  - `_get_best_model(comparison_df)`: Dapatkan model terbaik

### BaseBackbone (architectures/backbones/base.py)
- **Fungsi**: Kelas dasar untuk semua backbone network
- **Metode Utama**:
  - `get_output_channels()`: Dapatkan jumlah output channel
  - `get_output_shapes(input_size)`: Dapatkan dimensi output
  - `forward(x)`: Forward pass
  - `validate_output(features, expected_channels)`: Validasi output

### EfficientNetBackbone (architectures/backbones/efficientnet.py)
- **Fungsi**: Backbone EfficientNet untuk YOLOv5
- **Metode Utama**:
  - `forward(x)`: Forward pass dengan ekstraksi fitur dan adaptasi channel
  - `get_output_channels()`: Dapatkan jumlah output channel
  - `get_output_shapes(input_size)`: Dapatkan dimensi output feature maps

### CSPDarknet (architectures/backbones/cspdarknet.py)
- **Fungsi**: CSPDarknet backbone untuk YOLOv5
- **Metode Utama**:
  - `forward(x)`: Forward pass, mengembalikan feature maps
  - `get_output_channels()`: Dapatkan jumlah output channel
  - `get_output_shapes(input_size)`: Dapatkan dimensi output feature maps
  - `load_weights(state_dict, strict)`: Load state dictionary dengan validasi

### FeatureProcessingNeck (architectures/necks/fpn_pan.py)
- **Fungsi**: Neck untuk mengkombinasikan FPN dan PAN
- **Metode Utama**:
  - `forward(features)`: Forward pass FPN-PAN

### DetectionHead (architectures/heads/detection_head.py)
- **Fungsi**: Detection Head untuk YOLOv5 dengan dukungan multi-layer
- **Metode Utama**:
  - `forward(features)`: Forward pass detection head
  - `get_config()`: Dapatkan konfigurasi detection head

### YOLOLoss (components/losses.py)
- **Fungsi**: YOLOv5 Loss Function dengan CIoU
- **Metode Utama**:
  - `forward(predictions, targets)`: Hitung loss untuk prediksi dan target
  - `_build_targets(pred, targets, layer_idx)`: Build targets untuk satu skala