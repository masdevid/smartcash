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
└── layer_config.py            # LayerConfigManager: Konfigurasi layer deteksi
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

## 4. Domain Detection (Diperbarui)

```
smartcash/detection/
├── __init__.py                 # Ekspor komponen deteksi
├── detector.py                 # Detector: Koordinator utama deteksi
├── services/
│   ├── __init__.py
│   ├── inference/              # Layanan inferensi
│   │   ├── __init__.py
│   │   ├── inference_service.py # InferenceService: Layanan inferensi
│   │   ├── optimizers.py       # ModelOptimizer: Optimasi model
│   │   ├── batch_processor.py  # BatchProcessor: Processor batch gambar
│   │   └── accelerator.py      # HardwareAccelerator: Akselerator hardware
│   ├── postprocessing/         # Layanan pasca-inferensi
│   │   ├── __init__.py
│   │   ├── nms_processor.py    # NMSProcessor: Non-Maximum Suppression
│   │   ├── confidence_filter.py # ConfidenceFilter: Filter confidence
│   │   ├── bbox_refiner.py     # BBoxRefiner: Perbaikan bounding box
│   │   └── result_formatter.py # ResultFormatter: Format hasil deteksi
│   ├── evaluation/             # Layanan evaluasi model
│   │   ├── __init__.py
│   │   ├── metrics_calculator.py # MetricsCalculator: Penghitung metrik
│   │   ├── evaluator.py        # DetectionEvaluator: Evaluator model
│   │   └── benchmark.py        # DetectionBenchmark: Benchmark kinerja
│   └── visualization_adapter.py # DetectionVisualizationAdapter: Adapter untuk visualisasi dari domain model
├── models/
│   ├── __init__.py
│   ├── yolov5_model.py         # SmartCashYOLOv5: Model YOLOv5 dengan EfficientNet
│   ├── efficientnet_backbone.py # EfficientNetBackbone: Backbone EfficientNet
│   ├── fpn_pan_neck.py         # FPN_PAN_Neck: Neck model (FPN+PAN)
│   ├── detection_head.py       # DetectionHead: Head untuk deteksi objek
│   └── anchors.py              # AnchorGenerator: Generator anchors
├── utils/
│   ├── __init__.py
│   ├── preprocess/             # Utilitas preprocessing
│   │   ├── __init__.py
│   │   ├── image_transform.py  # ImageTransformer: Transformer gambar
│   │   ├── normalization.py    # ImageNormalizer: Normalisasi gambar
│   │   └── augmentation.py     # RealTimeAugmentation: Augmentasi real-time
│   ├── postprocess/            # Utilitas postprocessing
│   │   ├── __init__.py
│   │   ├── bbox_utils.py       # Utilitas bbox (konversi, scale, clip)
│   │   ├── nms_utils.py        # Utilitas NMS (standard, weighted, class-agnostic)
│   │   └── score_utils.py      # Utilitas score (filter, sigmoid)
│   └── optimization/           # Utilitas optimasi
│       ├── __init__.py
│       ├── weight_quantization.py # Fungsi kuantisasi bobot dan aktivasi
│       ├── pruning.py          # Fungsi pruning model
│       └── memory_optimization.py # Fungsi optimasi memori
└── adapters/
    ├── __init__.py
    ├── onnx_adapter.py         # ONNXModelAdapter: Adapter untuk model ONNX
    ├── torchscript_adapter.py  # TorchScriptAdapter: Adapter untuk TorchScript
    ├── tensorrt_adapter.py     # TensorRTAdapter: Adapter untuk TensorRT
    └── tflite_adapter.py       # TFLiteAdapter: Adapter untuk TFLite
```

## 5. Domain Model (Diperbarui)

```
smartcash/model/
├── __init__.py                 # Ekspor komponen model
├── manager.py                  # ModelManager: Koordinator alur kerja model
├── manager_checkpoint.py       # ModelCheckpointManager: Integrasi dengan checkpoint service
├── services/
│   ├── __init__.py
│   ├── checkpoint/             # Manajemen checkpoint model
│   │   ├── __init__.py
│   │   ├── checkpoint_service.py # CheckpointService: Layanan checkpoint 
│   │   ├── local_storage.py    # LocalCheckpointStorage: Storage checkpoint lokal
│   │   ├── drive_storage.py    # DriveCheckpointStorage: Storage checkpoint Drive
│   │   ├── sync_storage.py     # CheckpointStorageSynchronizer: Sinkronisasi storage
│   │   └── cleanup.py          # CheckpointCleanup: Pembersihan checkpoint
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
│   └── yolov5_model.py         # YOLOv5Model: Model YOLOv5 terintegrasi
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
├── visualization/              # Visualisasi model (Diimplementasikan)
│   ├── __init__.py             # Ekspor komponen visualisasi
│   ├── base_visualizer.py      # VisualizationHelper: Kelas utilitas dasar visualisasi
│   ├── base_research_visualizer.py # BaseResearchVisualizer: Kelas dasar visualisasi penelitian
│   ├── metrics_visualizer.py   # MetricsVisualizer: Visualisasi metrik
│   ├── detection_visualizer.py # DetectionVisualizer: Visualisasi deteksi
│   ├── experiment_visualizer.py # ExperimentVisualizer: Visualisasi eksperimen
│   ├── scenario_visualizer.py  # ScenarioVisualizer: Visualisasi skenario
│   ├── research_visualizer.py  # ResearchVisualizer: Visualisasi penelitian
│   └── evaluation_visualizer.py # EvaluationVisualizer: Visualisasi evaluasi
└── exceptions.py               # Eksepsi khusus model (ModelError, etc.)
```

## 6. Domain UI (Tidak Berubah)

```
smartcash/ui/
├── __init__.py                 # Ekspor komponen UI
├── cells/                      # Notebook cells utama
│   ├── setup/                  
│   │   ├── cell_1_1_repository_clone.py      # Clone repositori
│   │   ├── cell_1_2_environment_config.py    # Konfigurasi lingkungan
│   │   └── cell_1_3_dependency_installation.py # Instalasi dependencies
│   │
│   ├── dataset/                
│   │   ├── cell_2_1_dataset_download.py      # Download dataset
│   │   ├── cell_2_2_preprocessing.py         # Preprocessing dataset
│   │   ├── cell_2_3_split_config.py          # Konfigurasi split
│   │   └── cell_2_4_augmentation.py          # Augmentasi dataset
│   │
│   ├── training_config/        
│   │   ├── cell_3_1_backbone_selection.py    # Pemilihan backbone
│   │   ├── cell_3_2_hyperparameters.py       # Setting hyperparameter
│   │   ├── cell_3_3_training_strategy.py     # Strategi training
│   │   └── cell_3_4_layer_config.py          # Konfigurasi layer deteksi
│   │
│   ├── training_execution/     
│   │   ├── cell_4_1_model_training.py        # Pelatihan model
│   │   ├── cell_4_2_performance_tracking.py  # Tracking performa
│   │   └── cell_4_3_checkpoint_management.py # Manajemen checkpoint
│   │
│   └── model_evaluation/       
│       ├── cell_5_1_performance_metrics.py   # Metrik performa
│       ├── cell_5_2_comparative_analysis.py  # Analisis komparatif
│       └── cell_5_3_visualization.py         # Visualisasi hasil
│
├── components/                 # Komponen UI reusable
│   ├── shared/                 
│   │   ├── layouts.py          # Layout komponen
│   │   ├── styles.py           # Style dan tema
│   │   └── validators.py       # Validator input
│   │
│   ├── dataset/                
│   │   ├── download.py         # Komponen download dataset
│   │   ├── preprocessing.py    # Komponen preprocessing
│   │   └── augmentation.py     # Komponen augmentasi
│   │
│   ├── training_config/        
│   │   ├── backbone_selection.py # Komponen pemilihan backbone
│   │   ├── hyperparameters.py  # Komponen setting hyperparameter
│   │   └── training_strategy.py  # Komponen strategi training
│   │
│   ├── training_execution/     
│   │   ├── model_training.py   # Komponen pelatihan model
│   │   ├── performance_tracking.py # Komponen tracking performa
│   │   └── checkpoint_management.py # Komponen manajemen checkpoint
│   │
│   └── model_evaluation/       
│       ├── performance_metrics.py # Komponen metrik performa
│       ├── comparative_analysis.py # Komponen analisis komparatif
│       └── visualization.py    # Komponen visualisasi hasil
│
└── handlers/                   # Handler logika UI
    ├── shared/                 
    │   ├── config_handler.py   # Handler konfigurasi
    │   ├── observer_handler.py # Handler observer
    │   └── error_handler.py    # Handler error
    │
    ├── dataset/                
    │   ├── download_handler.py # Handler download dataset
    │   ├── preprocessing_handler.py # Handler preprocessing
    │   └── augmentation_handler.py # Handler augmentasi
    │
    ├── training_config/        
    │   ├── backbone_handler.py # Handler backbone
    │   ├── hyperparameters_handler.py # Handler hyperparameter
    │   └── training_strategy_handler.py # Handler strategi training
    │
    ├── training_execution/     
    │   ├── model_training_handler.py # Handler pelatihan model
    │   ├── performance_tracking_handler.py # Handler tracking performa
    │   └── checkpoint_handler.py # Handler checkpoint
    │
    └── model_evaluation/       
        ├── performance_metrics_handler.py # Handler metrik performa
        ├── comparative_analysis_handler.py # Handler analisis komparatif
        └── visualization_handler.py # Handler visualisasi

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