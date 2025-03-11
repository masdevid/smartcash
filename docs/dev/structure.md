# managers/: Folder utama yang menyimpan semua manajer dan utilitas terkait.

## managers/config/: Submodul untuk mengelola konfigurasi berbagai komponen.

- `base_config_manager.py`: Kelas dasar untuk manajer konfigurasi.
- `data_config_manager.py`: Manajer konfigurasi untuk dataset dan preprocessing.
- `model_config_manager.py`: Manajer konfigurasi untuk model dan hyperparameter.
- `training_config_manager.py`: Manajer konfigurasi untuk pelatihan dan evaluasi.


## managers/logging/: Submodul untuk pencatatan log di berbagai tingkatan dan output.

- `base_logger.py`: Kelas dasar untuk logger.
- `file_logger.py`: Logger untuk menulis log ke file.
- `console_logger.py`: Logger untuk mencetak log ke konsol.
- `colab_logger.py`: Logger khusus untuk lingkungan Google Colab.


## managers/preprocessing/: Submodul untuk preprocessing data sebelum pelatihan.

- `base_preprocessor.py`: Kelas dasar untuk preprocessor.
- `image_preprocessor.py`: Preprocessor untuk data gambar.
- `label_preprocessor.py`: Preprocessor untuk data label.
- `augmentation_preprocessor.py`: Preprocessor untuk augmentasi data.

## managers/utils/: Submodul untuk fungsi utilitas yang sering digunakan.

- `file_utils.py`: Utilitas untuk operasi file dan direktori.
- `image_utils.py`: Utilitas untuk manipulasi dan pemrosesan gambar.
- `label_utils.py`: Utilitas untuk manipulasi dan pemrosesan label.
- `bbox_utils.py`: Utilitas untuk operasi kotak pembatas (bounding box).
- `visualization_utils.py`: Utilitas untuk visualisasi dan plot.
- `tf_utils.py`: Utilitas untuk operasi TensorFlow.
- `environment_manager.py`: Deteksi environment (Colab/Local) dan konfigurasi path
- `memory_optimizer.py`: Manajemen memori (GC, clear TF session, mixed precision)
- `constants.py`: Konstanta global (DIRECTORIES, IMAGE_TYPES, LAYER_CLASSES)
- `safe_threading.py`: Processor multithread aman (ThreadPool dengan exception handling)


## managers/data_manager/: Submodul untuk manajemen dataset dan preprocessing.
- `__init__.py`
- `base_data_manager.py`: Kelas dasar untuk manajer data.
- `dataset_loader.py`: Modul untuk memuat dataset dari berbagai sumber.
- `data_preprocessor.py`: Modul untuk preprocessing data sebelum pelatihan.
- `data_augmentation.py`: Modul untuk augmentasi data.
- `data_splitter.py`: Modul untuk membagi dataset menjadi set pelatihan, validasi, dan pengujian.
- `data_validator.py`: Modul untuk memvalidasi integritas dan kualitas dataset.
- `roboflow_downloader.py`: Unduh dataset dari Roboflow API
- `roboflow_zip_extractor.py`: Ekstrak dataset zip dari Roboflow
- `local_zip_extractor.py`: Ekstrak zip yang di-upload dari komputer lokal


## managers/model_manager/: Submodul untuk manajemen model dan hyperparameter.
- `__init__.py`
- `base_model_manager.py`: Kelas dasar untuk manajer model.
- `model_builder.py`: Modul untuk membangun arsitektur model.
- `model_loader.py`: Modul untuk memuat model pra-terlatih.
- `model_saver.py`: Modul untuk menyimpan dan menyimpan checkpoint model.
- `model_exporter.py`: Modul untuk mengekspor model ke format yang dapat digunakan.

## manager/training_manager/: Submodul untuk manajemen pelatihan model.
- `__init__.py`
- `base_training_manager.py`: Kelas dasar untuk manajer pelatihan.
- `training_pipeline.py`: Modul untuk mengelola pipeline pelatihan end-to-end.
- `training_callbacks.py`: Modul untuk mengelola callback selama pelatihan.
- `training_metrics.py`: Modul untuk menghitung dan melacak metrik pelatihan.
- `training_scheduler.py`: Modul untuk menjadwalkan tugas-tugas pelatihan.

## managers/evaluation_manager/: Submodul untuk manajemen evaluasi model.
- `__init__.py`
- `base_evaluation_manager.py`: Kelas dasar untuk manajer evaluasi.
- `evaluation_metrics.py`: Modul untuk menghitung metrik evaluasi.
- `evaluation_visualizer.py`: Modul untuk memvisualisasikan hasil evaluasi.
- `evaluation_reporter.py`: Modul untuk menghasilkan laporan evaluasi.

## managers/visualization_manager/: Submodul untuk manajemen visualisasi.
- `__init__.py`
- `base_visualization_manager.py`: Kelas dasar untuk manajer visualisasi.
- `data_visualization.py`: Modul untuk memvisualisasikan dataset dan distribusinya.
- `training_visualization.py`: Modul untuk memvisualisasikan metrik dan kemajuan pelatihan.
- `evaluation_visualization.py`: Modul untuk memvisualisasikan hasil evaluasi model.
- `inference_visualization.py`: Modul untuk memvisualisasikan hasil inferensi model.

## managers/augmentation_manager/: Submodul untuk manajemen augmentasi.
- `__init__.py`
- `base_augmentation_manager.py`: Kelas dasar untuk manajer augmentasi.
- `image_augmentation.py`: Modul untuk augmentasi gambar.
- `label_augmentation.py`: Modul untuk augmentasi label.
- `augmentation_pipeline.py`: Modul untuk mengelola pipeline augmentasi.
- `augmentation_policy.py`: Modul untuk menentukan kebijakan augmentasi.

## managers/cache_manager/: Submodul untuk manajemen cache.
- `__init__.py`
- `base_cache_manager.py`: Kelas dasar untuk manajer cache.
- `data_cache.py`: Modul untuk caching data.
- `model_cache.py`: Modul untuk caching model.
- `cache_policy.py`: Modul untuk menentukan kebijakan caching.
- `cache_utils.py`: Modul utilitas untuk operasi caching.

## managers/experiment_manager/: Submodul untuk manajemen eksperimen.
- `__init__.py`
- `base_experiment_manager.py`: Kelas dasar untuk manajer eksperimen.
- `experiment_tracker.py`: Modul untuk melacak dan mencatat eksperimen.
- `experiment_analyzer.py`: Modul untuk menganalisis hasil eksperimen.
- `experiment_visualizer.py`: Modul untuk memvisualisasikan hasil eksperimen.
- `experiment_reporter.py`: Modul untuk menghasilkan laporan eksperimen.

## managers/observer/: Submodul untuk pola observer
- `__init__.py`
- `observer_manager.py`: Implementasi pola Observer untuk event tracking
- `training_observer.py`: Observer khusus monitoring pelatihan

## managers/checkpoint/: Submodul untuk manajemen checkpoint
- `__init__.py`
- `checkpoint_manager.py`: Auto-save/load checkpoint dengan strategi versioning
- `best_checkpoint_selector.py`: Seleksi checkpoint berdasarkan metrik
- `colab_checkpoint_sync.py`: Sinkronisasi checkpoint dengan Google Drive

## managers/metrics/: Submodul untuk manajemen metrik
- `__init__.py`
- `base_metrics.py`: Template metrik kustom
- `training_metrics.py`: Metrik khusus pelatihan (accuracy, loss adaptif)
- `evaluation_metrics.py`: Metrik evaluasi model (mAP, IoU)
- `early_stopping.py`: EarlyStopping dengan multi-metrik (min/max mode)

# managers/ui/: Reorganisasi UI berdasarkan alur Colab Notebook

## managers/ui/core/: Komponen UI dasar dan utilitas
- `__init__.py`
- `ui_utils.py` - Template styling, layout manager, dan helper functions
- `base_components.py` - Komponen UI dasar (progress bars, cards, sections)
- `event_dispatcher.py` - Sistem event handling terpusat

## managers/ui/1_project_setup/: UI untuk Section 1 - Setup Proyek
- `1.1_repository_clone.py` - UI untuk clone git & yolov5
- `1.2_environment_config.py` - Environment switcher (Colab/Local)
- `1.3_dependency_manager.py` - Dependency checker & installer

## managers/ui/2_dataset_prep/: UI untuk Section 2 - Persiapan Data
- `2.1_roboflow_downloader.py` - Integrasi Roboflow API + download progress
- `2.2_preprocessing_controls.py` - Konfigurasi preprocessing steps
- `2.3_split_configurator.py` - UI untuk train/val/test split
- `2.4_augmentation_pipeline.py` - Live preview augmentasi

## managers/ui/3_training_config/: UI untuk Section 3 - Konfigurasi Training
- `3.1_backbone_selector.py` - Model zoo browser dengan pretrained weights
- `3.2_hyperparameter_tuner.py` - Interactive sliders & presets
- `3.3_layer_customizer.py` - Layer selection
- `3.4_strategy_selector.py` - Distributed training options

## managers/ui/4_training_exec/: UI untuk Section 4 - Eksekusi Training
- `4.1_training_launcher.py` - Start/stop/pause controls
- `4.2_performance_dashboard.py` - Live metrics dashboard
- `4.3_checkpoint_browser.py` - Versioning dan restore interface
- `4.4_realtime_visualizer.py` - Grafik interaktif (opsional)

## managers/ui/5_evaluation/: UI untuk Section 5 - Evaluasi Model
- `5.1_metric_analyzer.py` - Comparative metrics table
- `5.2_statistical_insights.py` - Distribusi error analysis
- `5.3_ab_testing_ui.py` - Model comparison A/B testing
- `5.4_visualization_suite.py` - Confusion matrix, ROC curves

## managers/ui/6_prediction/: UI untuk Section 6 - Inferensi Real-time (opsional)
- `6.1_data_source_selector.py` - Upload vs Webcam toggle
- `6.2_model_selector.py` - Model version picker
- `6.3_realtime_monitor.py` - FPS counter & resource usage
- `6.4_export_interface.py` - Ekspor hasil ke berbagai format

Identify the root problem and resolve current issues in data preparation
- data_downloader:
   `dataset_download - INFO - ✅ ✅ Dataset smartcash-wo2us/rupiah-emisi-2022:3 berhasil didownload ke /content/data/roboflow_smartcash-wo2us_rupiah-emisi-2022_3` succesfully return the path but it actually hasn't downloaded any file yet! Directory checking must be done after the download is completed.
- upload zip from local doesn't seems to work!
ui_handler/preprocessing issue:
- cleanup preprocessed file removing original dataset instead of preprocessed images. Does preprocessing process is replacing original dataset instead of putting processed image into somewhere else?