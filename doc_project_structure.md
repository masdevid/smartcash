# Struktur Project SmartCash

## Dokumentasi Struktur Lengkap Project SmartCash

SmartCash adalah sistem deteksi nilai mata uang Rupiah menggunakan YOLOv5 dengan integrasi backbone EfficientNet-B4. Proyek ini didesain dengan pendekatan modular, mengikuti prinsip object-oriented programming dan design patterns.

```
smartcash/
├── app.py                     # Entry point aplikasi web (Gradio)
├── cli.py                     # Command Line Interface
│
├── config/                    # Pengelolaan konfigurasi
│   ├── __init__.py
│   ├── config_manager.py      # Manager untuk konfigurasi terpusat
│   └── base_config.yaml       # Konfigurasi default
│
├── exceptions/                # Pengelolaan error
│   ├── __init__.py
│   ├── base.py                # Exception dasar
│   ├── factory.py             # Factory untuk pembuatan error
│   └── handler.py             # Handler untuk pengelolaan error
│
├── factories/                 # Factory pattern
│   ├── __init__.py
│   ├── dataset_component_factory.py  # Factory untuk komponen dataset
│   ├── model_component_factory.py    # Factory untuk komponen model
│   └── training_component_factory.py # Factory untuk komponen training
│
├── handlers/                  # Handler untuk operasi kompleks
│   ├── __init__.py
│   │
│   ├── dataset/               # Pengelolaan dataset multilayer
│   │   ├── __init__.py
│   │   ├── dataset_manager.py        # Entry point untuk dataset
│   │   ├── facades/                  # Facades untuk fungsi spesifik
│   │   │   ├── __init__.py
│   │   │   ├── dataset_base_facade.py
│   │   │   ├── data_loading_facade.py
│   │   │   ├── data_processing_facade.py
│   │   │   ├── data_operations_facade.py
│   │   │   ├── visualization_facade.py
│   │   │   ├── dataset_explorer_facade.py
│   │   │   └── pipeline_facade.py
│   │   ├── multilayer/               # Komponen dataset multilayer
│   │   │   ├── __init__.py
│   │   │   ├── multilayer_dataset_base.py
│   │   │   ├── multilayer_dataset.py
│   │   │   └── multilayer_label_handler.py
│   │   ├── core/                     # Komponen inti
│   │   │   ├── __init__.py
│   │   │   ├── dataset_loader.py
│   │   │   ├── dataset_downloader.py
│   │   │   ├── dataset_transformer.py
│   │   │   ├── dataset_validator.py
│   │   │   ├── dataset_augmentor.py
│   │   │   ├── dataset_balancer.py
│   │   │   └── download_manager.py
│   │   ├── operations/               # Operasi pada dataset
│   │   │   ├── __init__.py
│   │   │   ├── dataset_split_operation.py
│   │   │   ├── dataset_merge_operation.py
│   │   │   └── dataset_reporting_operation.py
│   │   ├── explorers/                # Eksplorasi dataset
│   │   │   ├── __init__.py
│   │   │   ├── base_explorer.py
│   │   │   ├── validation_explorer.py
│   │   │   ├── class_explorer.py
│   │   │   ├── layer_explorer.py
│   │   │   ├── image_size_explorer.py
│   │   │   └── bbox_explorer.py
│   │   ├── integration/              # Adapter untuk integrasi
│   │   │   ├── __init__.py
│   │   │   ├── validator_adapter.py
│   │   │   └── colab_drive_adapter.py
│   │   └── visualizations/           # Visualisasi dataset
│   │       ├── __init__.py
│   │       ├── visualization_base.py
│   │       ├── heatmap/
│   │       │   ├── __init__.py
│   │       │   ├── spatial_density_heatmap.py
│   │       │   ├── class_density_heatmap.py
│   │       │   └── size_distribution_heatmap.py
│   │       └── sample/
│   │           ├── __init__.py
│   │           ├── sample_grid_visualizer.py
│   │           └── annotation_visualizer.py
│   │
│   ├── checkpoint/            # Pengelolaan checkpoint model
│   │   ├── __init__.py
│   │   ├── checkpoint_manager.py     # Entry point minimal (facade)
│   │   ├── checkpoint_loader.py      # Loading checkpoint model
│   │   ├── checkpoint_saver.py       # Penyimpanan checkpoint model
│   │   ├── checkpoint_finder.py      # Pencarian checkpoint
│   │   ├── checkpoint_history.py     # Pengelolaan riwayat training
│   │   └── checkpoint_utils.py       # Utilitas umum
│   │
│   ├── detection/             # Proses deteksi mata uang
│   │   ├── __init__.py
│   │   ├── detection_manager.py      # Entry point minimal (facade)
│   │   ├── core/                     # Komponen inti deteksi
│   │   │   ├── __init__.py
│   │   │   ├── detector.py
│   │   │   ├── preprocessor.py
│   │   │   └── postprocessor.py
│   │   ├── strategies/               # Strategi-strategi deteksi
│   │   │   ├── __init__.py
│   │   │   ├── base_strategy.py
│   │   │   ├── image_strategy.py
│   │   │   └── directory_strategy.py
│   │   ├── pipeline/                 # Pipeline deteksi
│   │   │   ├── __init__.py
│   │   │   ├── base_pipeline.py
│   │   │   ├── detection_pipeline.py
│   │   │   └── batch_pipeline.py
│   │   ├── integration/              # Adapter untuk integrasi
│   │   │   ├── __init__.py
│   │   │   ├── model_adapter.py
│   │   │   └── visualizer_adapter.py
│   │   ├── output/                   # Pengelolaan output
│   │   │   ├── __init__.py
│   │   │   └── output_manager.py
│   │   └── observers/                # Observer pattern
│   │       ├── __init__.py
│   │       ├── base_observer.py
│   │       ├── progress_observer.py
│   │       └── metrics_observer.py
│   │
│   ├── evaluation/            # Evaluasi model
│   │   ├── __init__.py
│   │   ├── evaluation_manager.py     # Entry point sebagai facade
│   │   ├── core/                     # Komponen inti evaluasi
│   │   │   ├── __init__.py
│   │   │   ├── evaluation_component.py
│   │   │   ├── model_evaluator.py
│   │   │   └── report_generator.py
│   │   ├── pipeline/                 # Pipeline dan workflow
│   │   │   ├── __init__.py
│   │   │   ├── base_pipeline.py
│   │   │   ├── evaluation_pipeline.py
│   │   │   ├── batch_evaluation_pipeline.py
│   │   │   └── research_pipeline.py
│   │   ├── integration/              # Adapter untuk integrasi
│   │   │   ├── __init__.py
│   │   │   ├── metrics_adapter.py
│   │   │   ├── model_manager_adapter.py
│   │   │   ├── dataset_adapter.py
│   │   │   ├── checkpoint_manager_adapter.py
│   │   │   ├── visualization_adapter.py
│   │   │   └── adapters_factory.py
│   │   └── observers/                # Observer pattern
│   │       ├── __init__.py
│   │       ├── base_observer.py
│   │       ├── progress_observer.py
│   │       └── metrics_observer.py
│   │
│   ├── model/                 # Pengelolaan model
│   │   ├── __init__.py
│   │   ├── model_manager.py          # Entry point minimal (facade)
│   │   ├── core/                     # Komponen inti model
│   │   │   ├── __init__.py
│   │   │   ├── model_component.py
│   │   │   ├── model_factory.py
│   │   │   ├── backbone_factory.py
│   │   │   ├── optimizer_factory.py
│   │   │   ├── model_trainer.py
│   │   │   ├── model_evaluator.py
│   │   │   └── model_predictor.py
│   │   ├── experiments/              # Eksperimen dan riset
│   │   │   ├── __init__.py
│   │   │   ├── experiment_manager.py
│   │   │   └── backbone_comparator.py
│   │   ├── observers/                # Observer untuk monitoring
│   │   │   ├── __init__.py
│   │   │   ├── base_observer.py
│   │   │   ├── metrics_observer.py
│   │   │   └── colab_observer.py
│   │   ├── integration/              # Adapter untuk integrasi
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint_adapter.py
│   │   │   ├── metrics_adapter.py
│   │   │   ├── environment_adapter.py
│   │   │   ├── experiment_adapter.py
│   │   │   └── exporter_adapter.py
│   │   └── visualizations/           # Visualisasi training dan evaluasi
│   │       ├── __init__.py
│   │       ├── metrics_visualizer.py
│   │       └── comparison_visualizer.py
│   │
│   └── preprocessing/         # Preprocessing dataset
│       ├── __init__.py
│       ├── preprocessing_manager.py  # Entry point minimal (facade)
│       ├── core/                     # Komponen inti preprocessing
│       │   ├── __init__.py
│       │   ├── preprocessing_component.py
│       │   ├── validation_component.py
│       │   └── augmentation_component.py
│       ├── pipeline/                 # Pipeline dan workflow
│       │   ├── __init__.py
│       │   ├── preprocessing_pipeline.py
│       │   ├── validation_pipeline.py
│       │   └── augmentation_pipeline.py
│       ├── integration/              # Adapter untuk integrasi
│       │   ├── __init__.py
│       │   ├── validator_adapter.py
│       │   ├── augmentation_adapter.py
│       │   ├── cache_adapter.py
│       │   └── colab_drive_adapter.py
│       └── observers/                # Observer pattern untuk monitoring
│           ├── __init__.py
│           ├── base_observer.py
│           └── progress_observer.py
│
├── models/                    # Definisi model dan arsitektur
│   ├── __init__.py
│   ├── yolov5_model.py         # Model YOLOv5 dengan backbone fleksibel
│   ├── baseline.py             # Model baseline untuk SmartCash
│   ├── detection_head.py       # Detection head dengan opsi multilayer
│   ├── losses.py               # Custom loss functions
│   ├── backbones/              # Backbone architectures
│   │   ├── __init__.py
│   │   ├── base.py             # Base class untuk semua backbones
│   │   ├── cspdarknet.py       # Implementasi CSPDarknet
│   │   └── efficientnet.py     # Adaptasi EfficientNet untuk YOLOv5
│   └── necks/                  # Feature processing
│       ├── __init__.py
│       └── fpn_pan.py          # Feature Pyramid Network dan Path Aggregation Network
│
├── ui_components/             # Komponen UI
│   ├── __init__.py
│   ├── data_components.py
│   ├── dataset_components.py
│   ├── directory_components.py
│   ├── augmentation_components.py
│   ├── config_components.py
│   ├── model_components.py
│   ├── model_playground_components.py
│   ├── evaluation_components.py
│   ├── research_components.py
│   ├── training_components.py
│   └── repository_components.py
│
├── ui_handlers/               # Handler untuk UI
│   ├── __init__.py
│   ├── common_utils.py
│   ├── data_handlers.py
│   ├── dataset_handlers.py
│   ├── directory_handlers.py
│   ├── augmentation_handlers.py
│   ├── config_handlers.py
│   ├── model_handlers.py
│   ├── model_playground_handlers.py
│   ├── evaluation_handlers.py
│   ├── research_handlers.py
│   ├── model_training_handlers.py
│   ├── training_pipeline_handlers.py
│   ├── training_config_handlers.py
│   └── repository_handlers.py
│
└── utils/                     # Utilitas untuk berbagai fungsi
    ├── __init__.py
    ├── logger.py              # Menggantikan simple_logger.py dengan SmartCashLogger
    ├── coordinate_utils.py    # Utilitas koordinat dan bounding box
    ├── metrics.py             # Perhitungan metrik evaluasi model
    ├── config_manager.py      # Pengelolaan konfigurasi terpusat
    ├── environment_manager.py # Pengelolaan environment (Colab/lokal)
    ├── early_stopping.py      # Handler early stopping dengan perbaikan
    ├── experiment_tracker.py  # Pelacakan dan penyimpanan eksperimen training
    ├── layer_config_manager.py # Pengelolaan konfigurasi layer deteksi
    ├── memory_optimizer.py    # Optimasi penggunaan memori (GPU/CPU)
    ├── model_exporter.py      # Ekspor model ke format produksi
    ├── ui_utils.py            # Utilitas UI untuk notebook/Colab
    ├── augmentation/          # Augmentasi dataset
    │   ├── __init__.py
    │   ├── augmentation_base.py
    │   ├── augmentation_pipeline.py
    │   ├── augmentation_processor.py
    │   ├── augmentation_validator.py
    │   ├── augmentation_checkpoint.py
    │   └── augmentation_manager.py
    ├── cache/                 # Sistem caching
    │   ├── __init__.py
    │   ├── cache_manager.py
    │   ├── cache_index.py
    │   ├── cache_storage.py
    │   ├── cache_cleanup.py
    │   └── cache_stats.py
    ├── dataset/               # Validasi dan analisis dataset
    │   ├── __init__.py
    │   ├── enhanced_dataset_validator.py
    │   ├── dataset_validator_core.py
    │   ├── dataset_analyzer.py
    │   ├── dataset_fixer.py
    │   ├── dataset_cleaner.py
    │   └── dataset_utils.py
    ├── training/              # Pipeline training
    │   ├── __init__.py
    │   ├── training_pipeline.py
    │   ├── training_callback.py
    │   ├── training_metrics.py
    │   ├── training_epoch.py
    │   └── validation_epoch.py
    └── visualization/         # Visualisasi hasil
        ├── __init__.py
        ├── base.py
        ├── detection.py
        ├── metrics.py
        ├── research.py
        ├── research_utils.py
        ├── scenario_visualizer.py
        ├── experiment_visualizer.py
        ├── evaluation_visualizer.py
        ├── research_analysis.py
        └── analysis/
            ├── __init__.py
            ├── experiment_analyzer.py
            └── scenario_analyzer.py
```

## Fitur Utama

### 1. Deteksi Multilayer Mata Uang Rupiah

SmartCash mendukung deteksi dengan beberapa layer deteksi:
- **Banknote**: Deteksi mata uang kertas (nominal penuh)
- **Nominal**: Deteksi area nominal tertentu
- **Security**: Deteksi fitur keamanan (tanda tangan, text, benang pengaman)

### 2. Integrasi Backbone Fleksibel

- **EfficientNet-B4**: Adaptasi untuk YOLOv5 dengan channel mapping
- **CSPDarknet**: Backbone default YOLOv5 dengan dukungan auto-download

### 3. Feature Processing Network

- **FPN (Feature Pyramid Network)**: Koneksi top-down untuk fitur multiscale
- **PAN (Path Aggregation Network)**: Koneksi bottom-up untuk detail lokal

### 4. Manajemen Dataset

- **Validasi dataset**: Deteksi dan perbaikan otomatis dataset rusak
- **Augmentasi dataset**: Augmentasi dengan berbagai teknik (posisi, pencahayaan)
- **Analisis dataset**: Analisis distribusi kelas, layer, dan ukuran bounding box

### 5. Training & Evaluasi

- **Pipeline training**: Sistem callback untuk monitoring dan visualisasi
- **Evaluasi model**: Evaluasi performa model dengan berbagai metrik
- **Skenario penelitian**: Perbandingan backbone dan kondisi pengujian

### 6. Integrasi Google Colab

- **Deteksi otomatis**: Mendeteksi environment Colab
- **Google Drive**: Integrasi dengan Google Drive untuk penyimpanan model dan dataset
- **UI interaktif**: UI yang kompatibel dengan notebook

### 7. User Interface

- **Web UI (Gradio)**: Antarmuka web interaktif dengan multiple tabs
- **CLI**: Command Line Interface untuk operasi batch

## Pola Desain yang Digunakan

- **Facade Pattern**: Manager sebagai entry point yang menyembunyikan kompleksitas
- **Factory Pattern**: Pembuatan komponen dengan dependency injection
- **Strategy Pattern**: Implementasi berbeda untuk operasi yang sama
- **Adapter Pattern**: Integrasi antar komponen yang berbeda
- **Observer Pattern**: Monitoring progress dan metrics
- **Component Pattern**: Struktur hierarkis komponen
- **Pipeline Pattern**: Alur kerja yang modular

## Alur Kerja Utama

### 1. Setup Environment & Dataset

```python
# 1. Initialize environment
from smartcash.handlers.directory import DirectoryManager
dir_manager = DirectoryManager()
dir_manager.setup_environment(colab_mode=True)

# 2. Setup dataset
from smartcash.handlers.dataset import DatasetManager
dataset_manager = DatasetManager()
dataset_manager.download_dataset(source="roboflow")
```

### 2. Preprocessing & Augmentasi

```python
# 1. Validasi dataset
from smartcash.handlers.preprocessing import PreprocessingManager
preprocessing_manager = PreprocessingManager()
preprocessing_manager.validate_dataset(split="train", fix_issues=True)

# 2. Augmentasi dataset
preprocessing_manager.augment_dataset(
    split="train", 
    augmentation_types=["combined", "lighting"], 
    num_variations=3
)
```

### 3. Training Model

```python
# 1. Initialize model
from smartcash.handlers.model import ModelManager
model_manager = ModelManager()
model = model_manager.create_model(
    backbone_type="efficientnet_b4",
    pretrained=True
)

# 2. Training
results = model_manager.train_model(
    model=model,
    train_data=train_loader,
    val_data=val_loader,
    epochs=50
)
```

### 4. Evaluasi & Perbandingan

```python
# 1. Evaluasi model tunggal
from smartcash.handlers.evaluation import EvaluationManager
eval_manager = EvaluationManager()
results = eval_manager.evaluate_model(model_path="best.pt", dataset="test")

# 2. Perbandingan backbone
research_results = eval_manager.evaluate_research_scenarios(
    backbones=["efficientnet_b4", "cspdarknet"],
    scenarios=["normal", "low_light", "rotation"]
)
```

### 5. Deteksi & Visualisasi

```python
# 1. Deteksi mata uang
from smartcash.handlers.detection import DetectionManager
detection_manager = DetectionManager()
results = detection_manager.detect(
    source="image.jpg",
    conf_threshold=0.25,
    visualize=True
)

# 2. Export model
model_manager.export_model(
    model_path="best.pt",
    format="onnx",
    input_shape=(640, 640)
)
```

## Pengembangan

SmartCash dikembangkan dengan fokus pada:

1. **Modularitas**: Komponen terpisah dengan tanggung jawab yang jelas
2. **Fleksibilitas**: Dukungan untuk berbagai backbone dan skenario
3. **Performa**: Optimasi untuk training dan inferensi
4. **Usability**: Antarmuka yang konsisten dan mudah digunakan
5. **Integrasi**: Dukungan untuk environment berbeda (local, Colab)

Beberapa cara untuk berkontribusi pada pengembangan:

- Implementasi backbone baru (misal ResNet, MobileNet)
- Evaluasi performa pada dataset mata uang lain
- Optimasi untuk inferensi pada perangkat edge/mobile
- Pengembangan antarmuka deployment