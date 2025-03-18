# SmartCash v2 - Struktur Proyek yang Diperbarui

## 1. Domain Common (Tidak Berubah)

```
smartcash/common/
├── __init__.py                # Ekspor utilitas umum dengan __all__
├── visualization/             # Visualisasi
│   ├── __init__.py            # Ekspor komponen visualisasi
│   ├── core/                  # Core visualisasi
│   │   ├── __init__.py        # Ekspor komponen core
│   │   └── visualization_base.py # Base class untuk visualisasi
│   └── helpers/               # Komponen visualisasi
│       ├── __init__.py        # Ekspor komponen helper visualisasi
│       ├── chart_helper.py    # ChartHelper: Visualisasi chart
│       ├── color_helper.py    # ColorHelper: Visualisasi warna
│       ├── annotation_helper.py # AnnotationHelper: Visualisasi anotasi
│       ├── export_helper.py   # ExportHelper: Export visualisasi
│       ├── layout_helper.py   # LayoutHelper: Layout visualisasi
│       └── style_helper.py    # StyleHelper: Styling visualisasi
├── interfaces/                # Abstract interfaces
│   ├── __init__.py            # Ekspor interfaces
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

## 2. Domain Components (Tidak Berubah)

```
smartcash/components/
├── __init__.py                 # Ekspor komponen reusable
├── observer/                   # Observer pattern
│   ├── __init__.py             # Ekspor komponen observer pattern
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
    ├── __init__.py             # Ekspor komponen cache
    ├── cleanup_cache.py        # CacheCleanup: Pembersihan cache otomatis
    ├── indexing_cache.py       # CacheIndex: Pengindeksan cache
    ├── manager_cache.py        # CacheManager: Manager cache utama
    ├── stats_cache.py          # CacheStats: Statistik cache
    └── storage_cache.py        # CacheStorage: Penyimpanan cache
```

## 3. Domain Dataset (Tidak Berubah)

```
smartcash/dataset/
├── __init__.py                 # Ekspor komponen dataset
├── manager.py                  # DatasetManager: Koordinator alur kerja dataset
├── services/
│   ├── __init__.py
│   ├── loader/                 # Layanan loading dataset
│   │   ├── __init__.py
│   │   ├── dataset_loader.py   # DatasetLoader: Loading dataset dari disk
│   │   ├── multilayer_loader.py # MultilayerLoader: Loader untuk dataset multilayer
│   │   ├── cache_manager.py    # DatasetCacheManager: Cache untuk dataset
│   │   └── batch_generator.py  # BatchGenerator: Generator batch data
│   │   ├── preprocessed_dataset_loader.py # PreprocessedDatasetLoader: Loading dataset hasil preprocessing
│   ├── validator/              # Layanan validasi dataset
│   │   ├── __init__.py
│   │   ├── dataset_validator.py # DatasetValidator: Validasi dataset utama
│   │   ├── label_validator.py  # LabelValidator: Validasi file label
│   │   ├── image_validator.py  # ImageValidator: Validasi gambar
│   │   └── fixer.py            # DatasetFixer: Perbaikan dataset
│   ├── preprocessor/           # Layanan preprocessing dataset
│   │   ├── __init__.py
│   │   ├── dataset_preprocessor.py   # Koordinator utama preprocessing dataset
│   │   ├── pipeline.py               # Pipeline transformasi preprocessing
│   │   ├── storage.py                # Pengelolaan file hasil preprocessing
│   │   └── cleaner.py                # Pembersih cache preprocessed
│   ├── augmentor/              # Layanan augmentasi dataset
│   │   ├── __init__.py
│   │   ├── augmentation_service.py # AugmentationService: Layanan augmentasi
│   │   ├── image_augmentor.py  # ImageAugmentor: Augmentasi gambar
│   │   ├── bbox_augmentor.py   # BBoxAugmentor: Augmentasi bounding box
│   │   └── pipeline_factory.py # AugmentationPipelineFactory: Factory pipeline
│   ├── downloader/             # Layanan download dataset
│   │   ├── __init__.py
│   │   ├── download_service.py # DownloadService: Service utama download
│   │   ├── roboflow_downloader.py # RoboflowDownloader: Download dari Roboflow
│   │   ├── download_validator.py # DownloadValidator: Validasi integritas download
│   │   └── file_processor.py   # FileProcessor: Pemrosesan file dataset
│   ├── explorer/               # Layanan eksplorasi dataset
│   │   ├── __init__.py
│   │   ├── explorer_service.py # ExplorerService: Layanan eksplorasi
│   │   ├── class_explorer.py   # ClassExplorer: Eksplorasi distribusi kelas
│   │   ├── layer_explorer.py   # LayerExplorer: Eksplorasi distribusi layer
│   │   ├── bbox_explorer.py    # BBoxExplorer: Eksplorasi bounding box
│   │   └── image_explorer.py   # ImageExplorer: Eksplorasi gambar
│   ├── balancer/               # Layanan balancing dataset
│   │   ├── __init__.py
│   │   ├── balance_service.py  # BalanceService: Layanan balancing
│   │   ├── undersampler.py     # Undersampler: Undersampling dataset
│   │   ├── oversampler.py      # Oversampler: Oversampling dataset
│   │   └── weight_calculator.py # WeightCalculator: Perhitungan bobot kelas
│   └── reporter/               # Layanan pelaporan dataset
│       ├── __init__.py
│       ├── report_service.py   # ReportService: Layanan pelaporan dataset
│       ├── metrics_reporter.py # MetricsReporter: Pelaporan metrik
│       ├── export_formatter.py # ExportFormatter: Format ekspor laporan
│       └── visualization_service.py # VisualizationService: Visualisasi metrik dan laporan
├── utils/
│   ├── __init__.py
│   ├── transform/              # Transformasi dataset
│   │   ├── __init__.py
│   │   ├── albumentations_adapter.py # AlbumentationsAdapter: Adapter Albumentations
│   │   ├── bbox_transform.py   # BBoxTransformer: Transformasi bounding box
│   │   ├── image_transform.py  # ImageTransformer: Transformasi gambar
│   │   ├── polygon_transform.py # PolygonTransformer: Transformasi polygon
│   │   └── format_converter.py # FormatConverter: Konversi format
│   ├── split/                  # Utilitas split dataset
│   │   ├── __init__.py
│   │   ├── dataset_splitter.py # DatasetSplitter: Split dataset
│   │   ├── merger.py           # DatasetMerger: Merge dataset
│   │   └── stratifier.py       # DatasetStratifier: Stratified split
│   ├── statistics/             # Statistik dataset
│   │   ├── __init__.py
│   │   ├── class_stats.py      # ClassStatistics: Statistik kelas
│   │   ├── image_stats.py      # ImageStatistics: Statistik gambar
│   │   └── distribution_analyzer.py # DistributionAnalyzer: Analisis distribusi
│   ├── file/                   # Pemrosesan file
│   │   ├── __init__.py
│   │   ├── file_processor.py   # FileProcessor: Processor file umum
│   │   ├── image_processor.py  # ImageProcessor: Processor gambar
│   │   └── label_processor.py  # LabelProcessor: Processor label
│   └── progress/               # Tracking progres
│       ├── __init__.py
│       ├── progress_tracker.py # ProgressTracker: Tracking progres
│       └── observer_adapter.py # ProgressObserver: Observer untuk progres
└── components/
    ├── __init__.py
    ├── datasets/               # Komponen dataset
    │   ├── __init__.py
    │   ├── base_dataset.py     # BaseDataset: Dataset dasar
    │   ├── multilayer_dataset.py # MultilayerDataset: Dataset multilayer
    │   └── yolo_dataset.py     # YOLODataset: Dataset format YOLO
    ├── geometry/               # Komponen geometri
    │   ├── __init__.py
    │   ├── polygon_handler.py  # PolygonHandler: Handler polygon
    │   ├── coord_converter.py  # CoordinateConverter: Konversi koordinat
    │   └── geometry_utils.py   # Utilitas geometri (IoU, area, clip)
    ├── labels/                 # Komponen label
    │   ├── __init__.py
    │   ├── label_handler.py    # LabelHandler: Handler label
    │   ├── multilayer_handler.py # MultilayerLabelHandler: Handler label multilayer
    │   └── format_converter.py # LabelFormatConverter: Konversi format label
    ├── samplers/               # Komponen sampler
    │   ├── __init__.py
    │   ├── balanced_sampler.py # BalancedBatchSampler: Sampler dengan balance kelas
    │   └── weighted_sampler.py # WeightedRandomSampler: Sampler dengan bobot
    └── collate/                # Komponen collate function
        ├── __init__.py
        ├── multilayer_collate.py # multilayer_collate_fn(): Collate untuk multilayer
        └── yolo_collate.py     # yolo_collate_fn(): Collate untuk YOLO
```

## 4. Domain Detection

```
ssmartcash/detection/
├── __init__.py                 # Ekspor komponen deteksi dengan __all__
├── detector.py                 # Detector: Koordinator utama proses deteksi
├── services/
│   ├── __init__.py             # Ekspor layanan detection dengan __all__
│   ├── inference/              # Layanan inferensi
│   │   ├── __init__.py         # Ekspor komponen inferensi dengan __all__
│   │   ├── inference_service.py # InferenceService: Koordinator inferensi model 
│   │   ├── accelerator.py      # HardwareAccelerator: Abstraksi hardware (CPU/GPU/TPU)
│   │   ├── batch_processor.py  # BatchProcessor: Processor batch gambar paralel
│   │   └── optimizers.py       # ModelOptimizer: Optimasi model (ONNX, TorchScript)
│   ├── postprocessing/         # Layanan pasca-inferensi
│   │   ├── __init__.py         # Ekspor komponen postprocessing dengan __all__
│   │   ├── postprocessing_service.py # PostprocessingService: Koordinator postprocessing
│   │   ├── confidence_filter.py # ConfidenceFilter: Filter confidence dengan threshold
│   │   ├── bbox_refiner.py     # BBoxRefiner: Perbaikan bounding box
│   │   └── result_formatter.py # ResultFormatter: Format hasil (JSON, CSV, YOLO, COCO)
│   └── visualization_adapter.py # Adapter visualisasi dari domain model
├── handlers/                   # Handler untuk berbagai skenario deteksi
│   ├── __init__.py             # Ekspor handler dengan __all__
│   ├── detection_handler.py    # DetectionHandler: Handler untuk deteksi gambar tunggal
│   ├── batch_handler.py        # BatchHandler: Handler untuk deteksi batch (folder/zip) 
│   ├── video_handler.py        # VideoHandler: Handler untuk deteksi video dan webcam
│   └── integration_handler.py  # IntegrationHandler: Integrasi dengan UI/API (async)
└── adapters/
    ├── __init__.py             # Ekspor adapter dengan __all__
    ├── onnx_adapter.py         # ONNXModelAdapter: Adapter untuk model ONNX
    └── torchscript_adapter.py  # TorchScriptAdapter: Adapter untuk model TorchScript
```

## 5. Domain Model (Diperbarui)

```
smartcash/model/
├── __init__.py                 # Ekspor komponen model deteksi objek
├── manager.py                  # ModelManager: Koordinator alur kerja model
├── manager_checkpoint.py       # ModelCheckpointManager: Integrasi dengan checkpoint service
├── services/
│   ├── __init__.py
│   ├── checkpoint/             # Manajemen checkpoint model
│   │   ├── __init__.py
│   │   └── checkpoint_service.py # CheckpointService: Layanan penyimpanan dan pemulihan checkpoint
│   ├── training/               # Layanan pelatihan model
│   │   ├── __init__.py
│   │   ├── core_training_service.py # TrainingService: Layanan training
│   │   ├── optimizer_training_service.py # OptimizerFactory: Factory untuk optimizer
│   │   ├── scheduler_training_service.py # SchedulerFactory: Factory untuk scheduler
│   │   ├── early_stopping_training_service.py # EarlyStoppingHandler: Early stopping
│   │   ├── warmup_scheduler_training_service.py # CosineDecayWithWarmup: Scheduler dengan warmup
│   │   ├── callbacks_training_service.py # TrainingCallbacks: Callback untuk training
│   │   └── experiment_tracker_training_service.py # ExperimentTracker: Tracking eksperimen
│   ├── evaluation/             # Layanan evaluasi model
│   │   ├── __init__.py
│   │   ├── core_evaluation_service.py # EvaluationService: Layanan evaluasi
│   │   ├── metrics_evaluation_service.py # MetricsComputation: Perhitungan metrik
│   │   └── visualization_evaluation_service.py # EvaluationVisualizer: Visualisasi hasil
│   ├── prediction/             # Layanan prediksi model
│   │   ├── __init__.py
│   │   ├── core_prediction_service.py # PredictionService: Layanan prediksi
│   │   ├── batch_processor_prediction_service.py # BatchPredictionProcessor: Processor batch
│   │   ├── interface_prediction_service.py # PredictionInterface: Interface prediksi
│   │   └── postprocessing_prediction_service.py # Postprocessing prediksi
│   ├── postprocessing/         # Layanan postprocessing
│   │   ├── __init__.py
│   │   └── nms_processor.py    # NMSProcessor: Processor untuk Non-Maximum Suppression
│   ├── experiment/             # Manajemen eksperimen
│   │   ├── __init__.py
│   │   ├── experiment_service.py # ExperimentService: Layanan eksperimen
│   │   ├── data_manager.py     # ExperimentDataManager: Manager data eksperimen
│   │   └── metrics_tracker.py  # ExperimentMetricsTracker: Tracking metrik
│   └── research/               # Skenario penelitian
│       ├── __init__.py
│       ├── experiment_service.py # ExperimentService: Layanan eksperimen penelitian
│       ├── scenario_service.py # ScenarioService: Layanan skenario penelitian
│       ├── experiment_creator.py # ExperimentCreator: Pembuat eksperimen
│       ├── experiment_runner.py # ExperimentRunner: Pelaksana eksperimen
│       ├── experiment_analyzer.py # ExperimentAnalyzer: Analisis eksperimen
│       ├── parameter_tuner.py  # ParameterTuner: Tuning parameter
│       └── comparison_runner.py # ComparisonRunner: Perbandingan model
├── analysis/                   # Analisis model
│   ├── __init__.py
│   ├── experiment_analyzer.py  # ExperimentAnalyzer: Analisis eksperimen
│   └── scenario_analyzer.py    # ScenarioAnalyzer: Analisis skenario
├── config/                     # Konfigurasi model
│   ├── __init__.py
│   ├── model_config.py         # ModelConfig: Konfigurasi model dasar
│   ├── backbone_config.py      # BackboneConfig: Konfigurasi backbone
│   └── experiment_config.py    # ExperimentConfig: Konfigurasi eksperimen
├── utils/                      # Utilitas model
│   ├── __init__.py
│   ├── preprocessing_model_utils.py # ModelPreprocessor: Preprocesor untuk model
│   ├── validation_model_utils.py # ModelValidator: Validasi model
│   ├── research_model_utils.py # Fungsi utilitas untuk penelitian
│   └── metrics/                # Perhitungan metrik
│       ├── __init__.py
│       ├── core_metrics.py     # Fungsi metrik dasar (IoU, mAP)
│       ├── ap_metrics.py       # Fungsi metrik AP (Average Precision)
│       ├── nms_metrics.py      # Fungsi metrik NMS
│       └── metrics_calculator.py # MetricsCalculator: Kalkulator metrik terintegrasi
├── components/                 # Komponen model
│   ├── __init__.py
│   └── losses.py               # Loss functions (YOLO, BBOX, Classification)
├── models/                     # Model terintegrasi
│   ├── __init__.py
│   ├── yolov5_model.py         # YOLOv5Model: Model YOLOv5 terintegrasi
│   └── smartcash_yolov5.py     # SmartCashYOLOv5: Model YOLOv5 dengan EfficientNet backbone
├── architectures/              # Arsitektur model
│   ├── __init__.py
│   ├── backbones/              # Backbone networks
│   │   ├── __init__.py
│   │   ├── base.py             # BaseBackbone: Kelas dasar untuk backbone
│   │   ├── efficientnet.py     # EfficientNetBackbone: Backbone EfficientNet
│   │   └── cspdarknet.py       # CSPDarknet: Backbone CSPDarknet
│   ├── necks/                  # Neck networks
│   │   ├── __init__.py
│   │   └── fpn_pan.py          # FeatureProcessingNeck: FPN+PAN neck
│   └── heads/                  # Head networks
│       ├── __init__.py
│       └── detection_head.py   # DetectionHead: Head deteksi
└── visualization/              # Visualisasi model
    ├── __init__.py             # Ekspor komponen visualisasi
    ├── base_visualizer.py      # ModelVisualizationBase: Kelas utilitas dasar visualisasi
    ├── metrics_visualizer.py   # MetricsVisualizer: Visualisasi metrik
    ├── detection_visualizer.py # DetectionVisualizer: Visualisasi deteksi
    ├── evaluation_visualizer.py # EvaluationVisualizer: Visualisasi evaluasi
    └── research/               # Visualisasi penelitian 
        ├── __init__.py
        ├── base_research_visualizer.py # BaseResearchVisualizer: Kelas dasar visualisasi penelitian
        ├── experiment_visualizer.py # ExperimentVisualizer: Visualisasi eksperimen
        ├── scenario_visualizer.py  # ScenarioVisualizer: Visualisasi skenario
        └── research_visualizer.py  # ResearchVisualizer: Visualisasi penelitian
```

## 6. Domain UI (Tidak Berubah)

```
smartcash/ui/
├── __init__.py                 # Ekspor komponen UI utama
├── components/                 # Komponen UI yang dapat digunakan kembali
│   ├── __init__.py
│   ├── alerts.py               # Komponen alert dan status
│   ├── headers.py              # Komponen header dan judul
│   ├── helpers.py              # Fungsi helper untuk UI
│   ├── layouts.py              # Layout standar untuk widget
│   ├── metrics.py              # Komponen untuk menampilkan metrik
│   └── validators.py           # Validasi input dan form
│
├── handlers/                   # Handler untuk berbagai event dan error
│   ├── __init__.py
│   ├── error_handler.py        # Penanganan error di UI
│   └── observer_handler.py     # Handler untuk observer events
│
├── utils/                      # Utilitas umum untuk UI
│   ├── __init__.py
│   ├── cell_utils.py           # Utilitas untuk cell notebook
│   ├── constants.py            # Konstanta untuk UI (warna, ikon)
│   ├── file_utils.py           # Utilitas file untuk UI
│   ├── logging_utils.py        # Logging terintegrasi dengan UI
│   ├── ui_helpers.py           # Helper generik untuk UI
│   └── visualization_utils.py  # Utilitas visualisasi
│
├── setup/                      # Komponen UI untuk setup proyek
│   ├── __init__.py
│   ├── cell_1_1_repo_clone.py  # Cell untuk clone repository
│   ├── cell_1_2_env_config.py  # Cell konfigurasi environment
│   ├── cell_1_3_dependency_installer.py  # Cell instalasi dependencies
│   ├── dependency_installer_component.py
│   ├── dependency_installer_config.py
│   ├── dependency_installer_handler.py
│   ├── directory_handler.py
│   ├── drive_handler.py
│   ├── env_config.py
│   ├── env_config_component.py
│   ├── env_config_handler.py
│   └── env_detection.py
│
├── dataset/                    # Komponen UI untuk manajemen dataset
│   ├── __init__.py
│   ├── cell_2_1_dataset_download.py  # Cell download dataset
│   ├── dataset_download_component.py
│   ├── dataset_download_handler.py
│   ├── download_click_handler.py
│   ├── download_confirmation_handler.py
│   ├── download_initialization.py
│   ├── download_ui_handler.py
│   ├── local_upload_handler.py
│   ├── roboflow_download_handler.py
│   └── download_ui_handler.py
│
├── training_config/            # Komponen UI untuk konfigurasi training
│   ├── __init__.py
│   ├── cell_3_1_backbone_selection.py  # Cell seleksi backbone
│   ├── cell_3_2_hyperparameters.py     # Cell hyperparameter
│   ├── cell_3_3_training_strategy.py   # Cell strategi training
│   ├── backbone_selection_component.py
│   ├── backbone_selection_handler.py
│   ├── config_buttons.py
│   ├── config_handler.py
│   ├── hyperparameters_component.py
│   ├── hyperparameters_handler.py
│   ├── training_strategy_component.py
│   └── training_strategy_handler.py
│
├── training_execution/         # Komponen UI untuk eksekusi training
│   ├── __init__.py
│   ├── cell_4_1_model_training.py
│   ├── model_training_component.py
│   ├── model_training_handler.py
│   ├── performance_tracking_component.py
│   ├── performance_tracking_handler.py
│   ├── checkpoint_management_component.py
│   └── checkpoint_management_handler.py
│
└── evaluation/                 # Komponen UI untuk evaluasi model
    ├── __init__.py
    ├── cell_5_1_performance_metrics.py
    ├── cell_5_2_model_comparison.py
    ├── performance_metrics_component.py
    ├── performance_metrics_handler.py
    ├── model_comparison_component.py
    ├── model_comparison_handler.py
    ├── visualization_component.py
    └── visualization_handler.py

```

## 7. Root Structure (Tidak Berubah)

```
smartcash/
├── __init__.py                 # Deklarasi package dan versi
├── main.py                     # Entry point untuk aplikasi CLI
├── cli.py                      # Command Line Interface
├── common/                     # Domain common
├── components/                 # Domain components (observer & cache)
├── dataset/                    # Domain dataset
├── detection/                  # Domain detection
├── model/                      # Domain model
├── ui/                         # Domain UI (Baru)
│   ├── cells/                  # Notebook cells
│   ├── components/             # Komponen UI
│   └── handlers/               # Handler logika UI
├── configs/                    # Konfigurasi (Diperbarui)
│   ├── base_config.yaml        # Konfigurasi dasar
│   ├── training_config.yaml    # Konfigurasi training
│   ├── evaluation_config.yaml  # Konfigurasi evaluasi
│   ├── augmentation_config.yaml # Konfigurasi augmentasi
│   ├── preprocessing_config.yaml # Konfigurasi preprocessing
│   ├── dataset_config.yaml     # Konfigurasi dataset
│   └── colab_config.yaml       # Konfigurasi khusus Colab
├── tests/                      # Unit dan integration tests
│   ├── __init__.py
│   ├── test_dataset.py         # Test untuk dataset
│   ├── test_detection.py       # Test untuk detection
│   └── test_model.py           # Test untuk model
├── setup.py                    # Script instalasi package
├── requirements.txt            # Dependency requirements
├── LICENSE                     # Informasi lisensi
└── README.md                   # Dokumentasi utama project
```