# SmartCash Project Structure

## Directory Structure
```
smartcash/
├── configs/               # Konfigurasi proyek
│   ├── base_config.yaml
│   └── train_config_20250304_141112.yaml
├── data/                 # Data mentah dan terproses
├── datasets/             # Dataset yang digunakan
│   └── multilayer_dataset.py
├── docs/                 # Dokumentasi proyek
│   ├── dev/             # Dokumentasi pengembang
│   │   ├── CODE_STYLE.md
│   │   ├── CONTRIBUTING.md
│   │   ├── GIT_GUIDELINES.md
│   │   ├── GIT_WORKFLOW.md
│   │   └── TESTING.md
│   ├── technical/       # Dokumentasi teknis
│   │   ├── API.md
│   │   ├── ARSITEKTUR.md
│   │   ├── DATASET.md
│   │   ├── EVALUASI.md
│   │   ├── MODEL.md
│   │   └── README.md
│   ├── user_guide/      # Panduan pengguna
│   │   ├── CLI.md
│   │   ├── EVALUATION.md
│   │   ├── INSTALASI.md
│   │   ├── README.md
│   │   ├── TRAINING.md
│   │   └── TROUBLESHOOTING.md
│   └── dataset/         # Dokumentasi dataset
│       ├── ANOTASI.md
│       ├── PREPROCESSING.md
│       ├── README.md
│       └── ROBOFLOW.md
├── exceptions/           # Custom exception classes
├── handlers/             # Handler untuk berbagai tugas
│   ├── backbone_handler.py
│   ├── base_evaluation_handler.py
│   ├── checkpoint_handler.py
│   ├── data_manager.py
│   ├── dataset_cleanup.py
│   ├── detection_handler.py
│   ├── evaluation_handler.py
│   ├── evaluator.py
│   ├── model_handler.py
│   ├── multilayer_dataset_handler.py
│   ├── multilayer_handler.py
│   ├── research_scenario_handler.py
│   ├── roboflow_handler.py
│   ├── unified-preprocessing.py
│   └── __init__.py
├── models/               # Model definitions dan weights
│   ├── backbones/       # Model backbones
│   │   ├── base.py
│   │   ├── cspdarknet.py
│   │   ├── efficientnet.py
│   │   └── __init__.py
│   ├── necks/          # Model necks
│   │   ├── fpn_pan.py
│   │   └── __init__.py
│   ├── baseline.py
│   ├── detection_head.py
│   ├── losses.py
│   ├── yolov5_model.py
│   └── __init__.py
├── notebooks/            # Jupyter notebooks
│   ├── Cell 11 - Instalasi Paket.txt
│   ├── Cell 12 - Cek Instalasi.txt
│   ├── Cell 20 - Clone Repository.txt
│   ├── Cell 30 - Setup Direktori.txt
│   ├── Cell 40 - Konfigurasi Global.txt
│   ├── Cell 50 - Konfigurasi Pipeline.txt
│   ├── Cell 60 - Dataset.txt
│   ├── Cell 71 - Inisialisasi Data Handling.txt
│   ├── Cell 72 - Informasi Dataset.txt
│   ├── Cell 73 - Augmentasi Data.txt
│   ├── Cell 74 - Split Dataset.txt
│   ├── Cell 75 - Fungsi Utilitas_Data.txt
│   ├── Cell 81 - Model Initialization.txt
│   ├── Cell 82 - Model Visualizer.txt
│   ├── Cell 83 - Model Playground.txt
│   ├── Cell 84 - Model Checkpoints.txt
│   ├── Cell 85 - Model Optimization.txt
│   ├── Cell 86 - Model Exporter.txt
│   ├── Cell 91 - Training Pipeline.txt
│   ├── Cell 92 - Training Configuration.txt
│   ├── Cell 93 - Training Execution.txt
│   ├── Cell 94 - Model Evaluation.txt
│   └── Cell 95 - Research Evaluation.txt
├── pretrained/           # Model pre-trained
├── runs/                 # Output eksperimen dan training
├── tests/               # Unit tests dan integration tests
└── utils/               # Utility functions dan tools
    ├── coordinate_normalizer.py
    ├── debug_helper.py
    ├── early_stopping.py
    ├── enhanced-augmentation.py
    ├── enhanced-cache.py
    ├── enhanced-dataset-validator.py
    ├── experiment_tracker.py
    ├── layer-config-manager.py
    ├── logger.py
    ├── memory_optimizer.py
    ├── metrics.py
    ├── model_exporter.py
    ├── model_visualizer.py
    ├── optimized-augmentation.py
    ├── polygon_metrics.py
    ├── preprocessing.py
    ├── roboflow_downloader.py
    ├── simple_logger.py
    ├── training_pipeline.py
    ├── visualization.py
    └── __init__.py
```

## Class Analysis

### utils/model_visualizer.py

#### ModelVisualizer
Visualizer untuk hasil evaluasi model dengan grafik dan tabel informatif.

Methods:
- `__init__(logger: Optional[SmartCashLogger], output_dir: str)`: Inisialisasi visualizer
- `setup_style()`: Setup style untuk visualisasi yang konsisten
- `visualize_evaluation_results(results: Dict[str, Any], title: str, show_confusion_matrix: bool, show_class_metrics: bool, save_plots: bool)`: Visualisasikan hasil evaluasi model
- `_create_radar_chart(ax, class_metrics, class_names)`: Buat radar chart untuk metrik per kelas
- `_show_class_metrics(class_metrics, class_names, save_plots)`: Tampilkan metrik per kelas dalam bentuk tabel dan heatmap
- `_show_model_summary(results, metrics, save_plots)`: Tampilkan ringkasan informasi model
- `_show_sample_detections(results, save_plots)`: Tampilkan contoh hasil deteksi
- `visualize_comparison(results_list: List[Dict[str, Any]], model_names: List[str], title: str, save_plots: bool)`: Visualisasikan perbandingan beberapa model
- `_plot_model_comparison(comparison_df, save_plots)`: Buat plot perbandingan model
- `_show_comparison_analysis(comparison_df)`: Tampilkan analisis perbandingan model
- `_plot_scenario_comparison(scenario_results, save_plots)`: Plot perbandingan skenario penelitian

#### ResultVisualizer
Visualizer untuk hasil deteksi mata uang dengan tampilan yang informatif.

Methods:
- `__init__(output_dir: str, logger: Optional[SmartCashLogger])`: Inisialisasi visualizer
- `visualize_detections(image: np.ndarray, detections: List[Dict], show_confidence: bool, show_value: bool, filename: Optional[str])`: Visualisasikan hasil deteksi pada gambar
- `_create_grid(images: List[np.ndarray], grid_size: Tuple[int, int], title: str)`: Buat grid dari gambar

#### Standalone Functions:
- `plot_confusion_matrix(cm: np.ndarray, class_names: List[str], normalize: bool, title: str, cmap: str, figsize: Tuple[int, int])`: Plot confusion matrix dengan format yang ditingkatkan
- `plot_detections(image: np.ndarray, detections: List[Dict], class_colors: Optional[Dict], show_conf: bool, figsize: Tuple[int, int])`: Plot hasil deteksi pada gambar
- `plot_training_metrics(metrics: Dict[str, List], figsize: Tuple[int, int])`: Plot metrik training

### utils/visualization.py

#### Standalone Functions:
- `plot_detections(images: Union[np.ndarray, torch.Tensor, List], detections: List[Dict], class_names: List[str], conf_threshold: float, figsize: Tuple[int, int], max_images: int, title: str, save_path: Optional[str], show: bool)`: Visualisasi hasil deteksi pada gambar

Key Features:
- Support untuk batch processing
- Fleksibel input format (numpy array, torch tensor, atau list)
- Customizable visualization parameters
- Grid layout untuk multiple images
- Automatic color palette generation for classes 