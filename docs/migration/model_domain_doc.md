# SmartCash v2 - Model Core Reference Documentation

## 1. File Structure
```
smartcash/model/
├── __init__.py                 # Ekspor komponen model utama
├── manager.py                  # ModelManager: Koordinator utama
├── manager_checkpoint.py       # ModelCheckpointManager: Integrasi checkpoint
├── architectures/
│   ├── __init__.py             # Ekspor komponen arsitektur
│   ├── backbones/              # Implementasi backbone
│   │   ├── __init__.py         # Ekspor komponen backbone
│   │   ├── base.py             # BaseBackbone: Kelas dasar untuk backbone
│   │   ├── cspdarknet.py       # CSPDarknet: Backbone YOLOv5 
│   │   └── efficientnet.py     # EfficientNet: Backbone EfficientNet
│   ├── necks/                  # Implementasi feature fusion
│   │   ├── __init__.py         # Ekspor komponen neck
│   │   └── fpn_pan.py          # Feature Pyramid & Path Aggregation Networks
│   └── heads/                  # Implementasi detection head
│       ├── __init__.py         # Ekspor komponen head
│       └── detection_head.py   # DetectionHead: Deteksi multi-layer
├── components/
│   ├── __init__.py             # Ekspor komponen model
│   └── losses.py               # YOLOLoss: Fungsi loss yang ditingkatkan
├── config/
│   ├── __init__.py             # Ekspor komponen konfigurasi
│   ├── model_config.py         # ModelConfig: Konfigurasi dasar
│   ├── backbone_config.py      # BackboneConfig: Konfigurasi backbone
│   └── experiment_config.py    # ExperimentConfig: Konfigurasi eksperimen
├── services/                   # Implementasi services
│   ├── checkpoint/             # Implementasi checkpoint
│   │   ├── __init__.py         # Ekspor komponen checkpoint
│   │   └── checkpoint_service.py   # CheckpointService: Layanan checkpoint
│   ├── evaluation/             # Implementasi evaluasi
│   │   ├── __init__.py         # Ekspor komponen evaluasi
│   │   ├── core_evaluation_service.py   # CoreEvaluationService: Layanan evaluasi core
│   │   ├── metrics_evaluation_service.py   # MetricsEvaluationService: Layanan evaluasi metrik
│   │   └── visualization_evaluation_service.py   # VisualizationEvaluationService: Layanan evaluasi visualisasi
│   ├── experiment/             # Implementasi eksperimen
│   │   ├── __init__.py         # Ekspor komponen eksperimen
│   │   ├── data_manager.py     # DataManager: Manajemen data
│   │   ├── experiment_service.py   # ExperimentService: Layanan eksperimen
│   │   └── metrics_tracker.py  # MetricsTracker: Pengawas metrik
│   ├── postprocessing/         # Implementasi postprocessing
│   │   ├── __init__.py         # Ekspor komponen postprocessing
│   │   └── nms_processor.py    # NMSProcessor: Processor untuk Non-Maximum Suppression
│   ├── prediction/             # Implementasi prediksi
│   │   ├── __init__.py         # Ekspor komponen prediksi
│   │   ├── batch_processor_prediction_service.py   # BatchProcessorPredictionService: Layanan prediksi batch
│   │   ├── core_prediction_service.py   # CorePredictionService: Layanan prediksi core
│   │   ├── interface_prediction_service.py   # InterfacePredictionService: Layanan prediksi interface
│   │   └── postprocessing_prediction_service.py   # PostprocessingPredictionService: Layanan prediksi postprocessing
│   ├── research/               # Implementasi penelitian
│   │   ├── __init__.py         # Ekspor komponen penelitian
│   │   ├── comparison_runner.py   # ComparisonRunner: Runner untuk perbandingan
│   │   ├── experiment_analyzer.py   # ExperimentAnalyzer: Analisis eksperimen
│   │   ├── experiment_creator.py   # ExperimentCreator: Pembuatan eksperimen
│   │   ├── experiment_runner.py   # ExperimentRunner: Runner eksperimen
│   │   ├── experiment_service.py   # ExperimentService: Layanan eksperimen
│   │   ├── parameter_tuner.py   # ParameterTuner: Tuner parameter
│   │   └── scenario_service.py   # ScenarioService: Layanan skenario
│   └── training/                 # Implementasi training
│       ├── __init__.py         # Ekspor komponen training
│       ├── callbacks_training_service.py   # CallbacksTrainingService: Layanan callback training
│       ├── core_training_service.py   # CoreTrainingService: Layanan training core
│       ├── early_stopping_training_service.py   # EarlyStoppingTrainingService: Layanan early stopping
│       ├── experiment_tracker_training_service.py   # ExperimentTrackerTrainingService: Layanan tracker eksperimen
│       ├── optimizer_training_service.py   # OptimizerTrainingService: Layanan optimizer
│       ├── scheduler_training_service.py   # SchedulerTrainingService: Layanan scheduler
│       └── warmup_scheduler_training_service.py   # WarmupSchedulerTrainingService: Layanan warmup scheduler
├── utils/                      # Utilitas umum
│   ├── __init__.py             # Ekspor utilitas
│   ├── preprocessing_model_utils.py   # Utilitas preprocessing
│   ├── research_model_utils.py       # Utilitas penelitian
│   ├── validation_model_utils.py     # Utilitas validasi
│   └── metrics/                # Modul metrik
│       ├── __init__.py         # Ekspor metrik
│       ├── core_metrics.py     # Metrik inti
│       ├── ap_metrics.py       # Metrik Average Precision
│       ├── nms_metrics.py      # Metrik Non-Maximum Suppression
│       └── metrics_calculator.py     # Kalkulator metrik
├── visualization/              # Komponen visualisasi
│   ├── __init__.py             # Ekspor visualisasi
│   ├── base_visualizer.py      # Visualizer dasar
│   ├── detection_visualizer.py # Visualisasi deteksi
│   ├── evaluation_visualizer.py    # Visualisasi evaluasi
│   ├── metrics_visualizer.py   # Visualisasi metrik
│   └── research/               # Visualisasi penelitian
│       ├── __init__.py         # Ekspor visualisasi penelitian
│       ├── base_research_visualizer.py   # Visualizer penelitian dasar
│       ├── experiment_visualizer.py      # Visualisasi eksperimen
│       ├── research_visualizer.py        # Visualisator penelitian
│       └── scenario_visualizer.py        # Visualisasi skenario
├── analysis/                   # Komponen analisis
│   ├── __init__.py             # Ekspor analisis
│   ├── experiment_analyzer.py  # Analisis eksperimen
│   └── scenario_analyzer.py    # Analisis skenario
└── models/                     # Implementasi model
    └── yolov5_model.py         # YOLOv5Model: Implementasi model utama
```

## 2. Komponen Inti dan Pemetaan Metode

### 2.1 ModelManager (`manager.py`)
- **Fungsi**: Koordinator utama untuk pengelolaan model
- **Metode Utama**:
  - `__init__(config, model_type, logger)`: Inisialisasi manager dengan konfigurasi
  - `build_model()`: Bangun model dengan backbone, neck, dan head
  - `_build_backbone()`: Inisialisasi komponen backbone
  - `_build_neck()`: Inisialisasi komponen neck
  - `_build_head()`: Inisialisasi komponen head 
  - `_build_loss_function()`: Inisialisasi fungsi loss
  - `create_model(model_type, **kwargs)`: Factory method untuk pembuatan model
  - `get_config()`: Dapatkan konfigurasi model
  - `update_config(config_updates)`: Perbarui konfigurasi model
  - `set_checkpoint_service(service)`: Set layanan checkpoint
  - `set_training_service(service)`: Set layanan training
  - `set_evaluation_service(service)`: Set layanan evaluasi
  - `save_model(path)`: Simpan model ke file
  - `load_model(path)`: Muat model dari file
  - `train(*args, **kwargs)`: Latih model menggunakan layanan training
  - `evaluate(*args, **kwargs)`: Evaluasi model menggunakan layanan evaluasi
  - `predict(image, conf_threshold, iou_threshold)`: Lakukan deteksi pada gambar

### 2.2 ModelCheckpointManager (`manager_checkpoint.py`)
- **Fungsi**: Mengelola checkpoint model dan persistensi model
- **Metode Utama**:
  - `__init__(model_manager, checkpoint_dir, max_checkpoints, logger)`: Inisialisasi manager checkpoint
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint model
  - `load_checkpoint(path, model, optimizer, map_location)`: Muat checkpoint model
  - `get_best_checkpoint()`: Dapatkan path ke checkpoint terbaik
  - `get_latest_checkpoint()`: Dapatkan path ke checkpoint terbaru
  - `list_checkpoints(sort_by)`: Daftar semua checkpoint yang tersedia
  - `export_to_onnx(output_path, input_shape, opset_version, dynamic_axes)`: Ekspor model ke format ONNX

### 2.3 BaseBackbone (`architectures/backbones/base.py`)
- **Fungsi**: Kelas dasar abstrak untuk implementasi backbone
- **Metode Utama**:
  - `__init__()`: Inisialisasi backbone dasar
  - `forward(x)`: Metode forward pass abstrak
  - `get_info()`: Dapatkan informasi backbone
  - `load_state_dict_from_path(state_dict_path, strict)`: Muat state dict dari checkpoint
  - `freeze()`: Bekukan parameter backbone
  - `unfreeze()`: Buka pembekuan parameter backbone

### 2.4 CSPDarknet (`architectures/backbones/cspdarknet.py`)
- **Fungsi**: Backbone CSPDarknet untuk YOLOv5
- **Metode Utama**:
  - `__init__(pretrained, model_size, weights_path, fallback_to_local, pretrained_dir, logger)`: Inisialisasi CSPDarknet
  - `_setup_pretrained_model(weights_path, fallback_to_local)`: Setup model pretrained
  - `_setup_model_from_scratch()`: Setup model tanpa bobot pretrained
  - `_download_weights(url, output_path)`: Unduh bobot model
  - `_extract_backbone(model)`: Ekstrak backbone dari model YOLOv5
  - `_verify_model_structure()`: Verifikasi struktur model dengan dummy forward pass
  - `forward(x)`: Forward pass melalui CSPDarknet
  - `get_output_channels()`: Dapatkan channel output untuk setiap level feature
  - `get_output_shapes(input_size)`: Dapatkan dimensi spasial output
  - `load_weights(state_dict, strict)`: Muat bobot dengan validasi

### 2.5 EfficientNetBackbone (`architectures/backbones/efficientnet.py`)
- **Fungsi**: Backbone EfficientNet dengan adaptasi YOLOv5
- **Metode Utama**:
  - `__init__(pretrained, model_name, out_indices, use_attention, logger)`: Inisialisasi backbone EfficientNet
  - `get_output_channels()`: Dapatkan channel output untuk setiap level
  - `get_output_shapes(input_size)`: Dapatkan dimensi spasial output
  - `forward(x)`: Forward pass dengan ekstraksi fitur dan adaptasi

### 2.6 FeatureProcessingNeck (`architectures/necks/fpn_pan.py`)
- **Fungsi**: Fusi fitur menggunakan FPN dan PAN dengan blok residual
- **Metode Utama**:
  - `__init__(in_channels, out_channels, num_repeats, logger)`: Inisialisasi feature neck
  - `forward(features)`: Forward pass melalui FPN-PAN

### 2.7 FeaturePyramidNetwork (`architectures/necks/fpn_pan.py`)
- **Fungsi**: Pathway top-down untuk peningkatan fitur
- **Metode Utama**:
  - `__init__(in_channels, out_channels, num_repeats)`: Inisialisasi FPN
  - `forward(features)`: Forward pass untuk FPN

### 2.8 PathAggregationNetwork (`architectures/necks/fpn_pan.py`)
- **Fungsi**: Pathway bottom-up untuk peningkatan fitur
- **Metode Utama**:
  - `__init__(in_channels, out_channels, num_repeats)`: Inisialisasi PAN
  - `forward(fpn_features)`: Forward pass untuk PAN

### 2.9 DetectionHead (`architectures/heads/detection_head.py`)
- **Fungsi**: Detection head multi-layer untuk YOLO
- **Metode Utama**:
  - `__init__(in_channels, layers, num_anchors, logger)`: Inisialisasi detection head
  - `_build_single_head(in_ch, num_classes, num_anchors)`: Bangun detection head untuk satu skala
  - `_conv_block(in_ch, out_ch, kernel_size)`: Buat blok konvolusi dengan BN dan aktivasi
  - `forward(features)`: Forward pass detection head
  - `get_config()`: Dapatkan konfigurasi detection head

### 2.10 YOLOLoss (`components/losses.py`)
- **Fungsi**: Fungsi loss yang ditingkatkan untuk YOLOv5
- **Metode Utama**:
  - `__init__(num_classes, anchors, anchor_t, balance, box_weight, cls_weight, obj_weight, label_smoothing, eps, use_ciou, logger)`: Inisialisasi YOLOLoss
  - `forward(predictions, targets)`: Hitung loss untuk prediksi dan target
  - `_validate_inputs(predictions, targets)`: Validasi format input
  - `_standardize_targets(targets)`: Standarisasi format target
  - `_build_targets(pred, targets, layer_idx)`: Bangun target untuk satu skala
  - `compute_loss(predictions, targets, model, active_layers)`: Hitung combined loss untuk semua layer aktif

### 2.11 ModelConfig (`config/model_config.py`)
- **Fungsi**: Konfigurasi dasar untuk model SmartCash
- **Metode Utama**:
  - `__init__(config_path, **kwargs)`: Inisialisasi konfigurasi model
  - `_update_from_kwargs(kwargs)`: Perbarui konfigurasi dari keyword arguments
  - `_deep_update(target, source)`: Perbarui dictionary secara rekursif
  - `_validate_config()`: Validasi parameter konfigurasi
  - `load_from_yaml(yaml_path)`: Muat konfigurasi dari file YAML
  - `save_to_yaml(yaml_path)`: Simpan konfigurasi ke file YAML
  - `get(key, default)`: Dapatkan nilai konfigurasi dengan dukungan dot notation
  - `set(key, value)`: Set nilai konfigurasi dengan dukungan dot notation
  - `update(updates)`: Perbarui konfigurasi dengan dictionary baru

### 2.12 BackboneConfig (`config/backbone_config.py`)
- **Fungsi**: Konfigurasi untuk backbone network
- **Metode Utama**:
  - `__init__(model_config, backbone_type)`: Inisialisasi konfigurasi backbone
  - `get(key, default)`: Dapatkan nilai konfigurasi backbone
  - `is_efficientnet()`: Periksa apakah backbone adalah EfficientNet
  - `get_feature_channels()`: Dapatkan dimensi channel untuk setiap stage
  - `to_dict()`: Konversi konfigurasi ke dictionary

### 2.13 ExperimentConfig (`config/experiment_config.py`)
- **Fungsi**: Konfigurasi untuk eksperimen training
- **Metode Utama**:
  - `__init__(name, base_config, experiment_dir, **kwargs)`: Inisialisasi konfigurasi eksperimen
  - `_save_initial_config()`: Simpan konfigurasi awal
  - `log_metrics(epoch, train_loss, val_loss, test_loss, **additional_metrics)`: Catat metrik untuk epoch
  - `_save_metrics()`: Simpan metrik ke file
  - `load_metrics()`: Muat metrik dari file
  - `get_best_metrics()`: Dapatkan metrik terbaik berdasarkan validation loss
  - `save_model_checkpoint(model, optimizer, epoch, is_best)`: Simpan checkpoint model
  - `load_model_checkpoint(checkpoint_path)`: Muat checkpoint model
  - `generate_report(include_plots)`: Buat laporan eksperimen
  - `_generate_plots()`: Buat plot metrik
  - `compare_experiments(experiment_dirs, output_dir, include_plots)`: Bandingkan beberapa eksperimen

## 3. Komponen Tambahan

### 3.1 ChannelAttention (`architectures/backbones/efficientnet.py`) 
- **Fungsi**: Mekanisme channel attention untuk peningkatan fitur
- **Metode Utama**:
  - `__init__(channels, reduction_ratio)`: Inisialisasi channel attention
  - `forward(x)`: Terapkan mekanisme attention

### 3.2 FeatureAdapter (`architectures/backbones/efficientnet.py`)
- **Fungsi**: Adapter untuk memetakan fitur EfficientNet ke format YOLO
- **Metode Utama**:
  - `__init__(in_channels, out_channels, use_attention)`: Inisialisasi feature adapter
  - `forward(x)`: Adaptasi fitur

### 3.3 ConvBlock (`architectures/necks/fpn_pan.py`)
- **Fungsi**: Blok konvolusi dengan BatchNorm dan aktivasi
- **Metode Utama**:
  - `__init__(in_channels, out_channels, kernel_size, stride, padding)`: Inisialisasi ConvBlock
  - `forward(x)`: Forward pass

### 3.4 ResidualBlock (`architectures/necks/fpn_pan.py`)
- **Fungsi**: Blok residual untuk menjaga aliran informasi
- **Metode Utama**:
  - `__init__(channels)`: Inisialisasi ResidualBlock
  - `forward(x)`: Forward pass dengan residual connection

## 4. Konstanta Konfigurasi

### 4.1 Tipe Model (`manager.py`)
- `OPTIMIZED_MODELS`: Dictionary konfigurasi model
  - `yolov5s`: YOLOv5s dasar dengan CSPDarknet
  - `efficient_basic`: Model dasar tanpa optimasi
  - `efficient_optimized`: Model dengan EfficientNet-B4 dan FeatureAdapter
  - `efficient_advanced`: Model dengan semua optimasi
  - `efficient_experiment`: Model penelitian dengan konfigurasi kustom

### 4.2 Tipe Backbone (`manager.py`)
- `SUPPORTED_BACKBONES`: Dictionary tipe backbone yang didukung
  - Varian EfficientNet (B0-B5)
  - Varian CSPDarknet (S/M/L)

### 4.3 Layer Deteksi (`architectures/heads/detection_head.py`)
- `LAYER_CONFIG`: Dictionary konfigurasi layer deteksi
  - `banknote`: Deteksi uang kertas utama (7 kelas)
  - `nominal`: Deteksi area nominal (7 kelas)
  - `security`: Deteksi fitur keamanan (3 kelas)

### 4.4 Konfigurasi Backbone (`config/backbone_config.py`)
- `BACKBONE_CONFIGS`: Dictionary konfigurasi backbone
  - Dimensi channel dan parameter scaling untuk setiap backbone

## 5. Services Mapping

### 5.1 CheckpointService (services/checkpoint/checkpoint_service.py)

- **Fungsi**: Manajemen checkpoint model
- **Metode Utama**:
  - `__init__(checkpoint_dir, max_checkpoints)`: Inisialisasi layanan checkpoint
  - `save_checkpoint(model, path, optimizer, epoch, metadata, is_best)`: Simpan checkpoint model
  - `load_checkpoint(path, model, optimizer)`: Muat checkpoint model
  - `get_latest_checkpoint()`: Dapatkan path checkpoint terbaru
  - `get_best_checkpoint()`: Dapatkan path checkpoint terbai    k
  - `list_checkpoints(sort_by)`: Daftar semua checkpoint
  - `export_to_onnx(model, output_path)`: Ekspor model ke ONNX
  - `add_metadata(checkpoint_path, metadata)`: Tambah metadata ke checkpoint

### 5.2 EvaluationService (services/evaluation/core_evaluation_service.py)

- **Fungsi**: Evaluasi performa model
- **Metode Utama**:
  - `__init__(config, output_dir)`: Inisialisasi layanan evaluasi
  - `evaluate(model, dataloader, conf_thres, iou_thres)`: Evaluasi model keseluruhan
  - `evaluate_by_layer(model, dataloader)`: Evaluasi per layer model
  - `evaluate_by_class(model, dataloader)`: Evaluasi performa per kelas

### 5.3 ExperimentDataManager (services/experiment/data_manager.py)

- **Fungsi**: Manajemen dan pembagian dataset
- **Metode Utama**:
  - `__init__(dataset_path, batch_size)`: Inisialisasi manajer data
  - `load_dataset(dataset_path, transform)`: Muat dataset dari path
  - `create_data_splits(val_split, test_split)`: Buat pembagian data training/validasi/testing
  - `create_data_loaders(batch_size)`: Buat data loaders untuk training
  - `get_dataset_stats()`: Dapatkan statistik detail dataset

### 5.4 BatchPredictionProcessor (services/prediction/batch_processor_prediction_service.py)

- **Fungsi**: Proses prediksi batch
- **Metode Utama**:
  - `__init__(prediction_service, output_dir)`: Inisialisasi processor batch
  - `process_directory(input_dir, save_results)`: Proses seluruh direktori gambar
  - `process_files(files, save_results)`: Proses list file spesifik
  - `run_and_save(input_source, output_filename)`: Jalankan dan simpan prediksi

### 5.5 PredictionService (services/prediction/core_prediction_service.py)

- **Fungsi**: Layanan prediksi model
- **Metode Utama**:
  - `__init__(model, config)`: Inisialisasi layanan prediksi
  - `predict(images, return_annotated)`: Prediksi objek dalam gambar
  - `predict_from_files(image_paths)`: Prediksi dari path file gambar
  - `visualize_predictions(image, detections)`: Visualisasi hasil prediksi

### 5.6 ExperimentService (services/experiment/experiment_service.py)

- **Fungsi**: Manajemen dan eksekusi eksperimen
- **Metode Utama**:
  - `__init__(base_dir, config)`: Inisialisasi layanan eksperimen
  - `create_experiment(name, description)`: Buat eksperimen baru
  - `run_experiment(experiment, dataset_path)`: Jalankan eksperimen
  - `run_comparison_experiment(name, dataset_path)`: Bandingkan model
  - `run_parameter_tuning(name, dataset_path)`: Lakukan tuning parameter

### 5.7 TrainingService (services/training/core_training_service.py)

- **Fungsi**: Layanan training model
- **Metode Utama**:
  - `__init__(model, config, device)`: Inisialisasi layanan training
  - `train(train_loader, val_loader, callbacks)`: Latih model
  - `_train_epoch(train_loader)`: Proses training satu epoch
  - `_validate_epoch(val_loader)`: Proses validasi satu epoch
  - `resume_from_checkpoint(checkpoint_path)`: Lanjutkan training dari checkpoint

### 5.8 ParameterTuner (services/research/parameter_tuner.py)

- **Fungsi**: Otomatisasi pencarian hyperparameter optimal
- **Metode Utama**:
  - `__init__(base_dir, experiment_runner)`: Inisialisasi parameter tuner
  - `run_parameter_tuning(name, dataset_path)`: Jalankan tuning parameter
  - `_generate_param_combinations(param_grid)`: Generate kombinasi parameter

### 5.9 ExperimentAnalyzer (services/research/experiment_analyzer.py)

- **Fungsi**: Analisis dan perbandingan eksperimen
- **Metode Utama**:
  - `__init__(base_dir)`: Inisialisasi analyzer eksperimen
  - `get_experiment_results(experiment_id)`: Dapatkan hasil eksperimen spesifik
  - `list_experiments(filter_tags)`: Daftar semua eksperimen
  - `compare_experiments(experiment_ids)`: Bandingkan beberapa eksperimen
  - `analyze_experiment_performance(experiment_id)`: Analisis performa eksperimen detil

## 6. Visualization Components

### 6.1 BaseVisualizer (`visualization/base_visualizer.py`)
- **Fungsi**: Kelas dasar untuk visualisasi dengan fungsionalitas umum
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi base visualizer
  - `set_plot_style(style)`: Atur gaya plot matplotlib
  - `save_figure(fig, filepath, dpi, bbox_inches)`: Simpan figure matplotlib
  - `create_output_directory(output_dir)`: Buat direktori output

### 6.2 DetectionVisualizer (`visualization/detection_visualizer.py`)
- **Fungsi**: Visualisasi hasil deteksi objek dengan bounding box dan label
- **Metode Utama**:
  - `__init__(output_dir, class_colors, logger)`: Inisialisasi visualizer deteksi
  - `visualize_detection(image, detections, filename, conf_threshold, show_labels, show_conf, show_total, show_value)`: Visualisasikan deteksi pada gambar
  - `visualize_detections_grid(images, detections_list, title, filename, grid_size, conf_threshold)`: Visualisasikan multiple deteksi dalam grid
  - `calculate_denomination_total(detections)`: Hitung total nilai mata uang dari deteksi

### 6.3 EvaluationVisualizer (`visualization/evaluation_visualizer.py`)
- **Fungsi**: Visualisasi hasil evaluasi model
- **Metode Utama**:
  - `__init__(config, output_dir, logger)`: Inisialisasi visualizer evaluasi
  - `create_all_plots(metrics_data, prefix, **kwargs)`: Buat semua visualisasi yang tersedia
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize, **kwargs)`: Plot confusion matrix
  - `plot_map_f1_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan mAP dan F1 score
  - `plot_inference_time(metrics_df, prefix, **kwargs)`: Plot waktu inferensi
  - `plot_backbone_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan backbone
  - `plot_condition_comparison(metrics_df, prefix, **kwargs)`: Plot perbandingan kondisi
  - `plot_combined_heatmap(metrics_df, prefix, **kwargs)`: Plot heatmap kombinasi backbone dan kondisi

### 6.4 MetricsVisualizer (`visualization/metrics_visualizer.py`)
- **Fungsi**: Visualisasi metrik evaluasi model
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi metrics visualizer
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize, figsize, cmap)`: Plot confusion matrix
  - `plot_training_metrics(metrics, title, filename, figsize, include_lr)`: Plot metrik training
  - `plot_accuracy_metrics(metrics, title, filename, figsize)`: Plot metrik akurasi
  - `plot_model_comparison(comparison_data, metric_cols, title, filename, figsize)`: Plot perbandingan model
  - `plot_research_comparison(results_df, metric_cols, title, filename, figsize)`: Plot perbandingan skenario penelitian

### 6.5 BaseResearchVisualizer (`visualization/research/base_research_visualizer.py`)
- **Fungsi**: Kelas dasar untuk visualisasi penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi base research visualizer
  - `_create_styled_dataframe(df)`: Buat DataFrame dengan styling highlight
  - `_add_tradeoff_regions(ax)`: Tambahkan regions untuk visualisasi trade-off
  - `save_visualization(fig, filename)`: Simpan visualisasi ke file

### 6.6 ExperimentVisualizer (`visualization/research/experiment_visualizer.py`)
- **Fungsi**: Visualisasi hasil eksperimen model
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi experiment visualizer
  - `visualize_experiment_comparison(results_df, title, filename, highlight_best, figsize)`: Visualisasikan perbandingan eksperimen

### 6.7 ResearchVisualizer (`visualization/research/research_visualizer.py`)
- **Fungsi**: Visualisasi dan analisis penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi research visualizer
  - `visualize_experiment_comparison(results_df, title, filename, highlight_best, figsize)`: Visualisasikan perbandingan eksperimen
  - `visualize_scenario_comparison(results_df, title, filename, figsize)`: Visualisasikan perbandingan skenario penelitian

### 6.8 ScenarioVisualizer (`visualization/research/scenario_visualizer.py`)
- **Fungsi**: Visualisasi skenario penelitian
- **Metode Utama**:
  - `__init__(output_dir, logger)`: Inisialisasi scenario visualizer
  - `visualize_scenario_comparison(results_df, title, filename, figsize)`: Visualisasikan perbandingan skenario penelitian
  - `_filter_successful_scenarios(df)`: Filter skenario yang sukses
  - `_create_accuracy_plot(ax, df, metric_cols, backbone_col)`: Buat plot akurasi per skenario
  - `_create_inference_time_plot(ax, df, time_col, backbone_col)`: Buat plot waktu inferensi per skenario
  - `_create_backbone_comparison_plot(ax, df, metric_cols, backbone_col)`: Buat plot perbandingan backbone
  - `_create_condition_comparison_plot(ax, df, metric_cols, condition_col)`: Buat plot perbandingan kondisi

## 7. Analysis Components

### 7.1 ExperimentAnalyzer (`analysis/experiment_analyzer.py`)
- **Fungsi**: Analisis hasil eksperimen dan pemberian rekomendasi
- **Metode Utama**:
  - `analyze_experiment_results(df, metric_cols, time_col)`: Analisis hasil eksperimen
  - `_get_representative_metric(metric_cols)`: Pilih metrik representatif
  - `_find_model_column(df)`: Temukan kolom yang berisi nama model
  - `_identify_best_model(best_row, metric, model_col, idx)`: Identifikasi model terbaik
  - `_identify_fastest_model(fastest_row, time_col, model_col, idx)`: Identifikasi model tercepat
  - `_calculate_metrics_statistics(df, metric_cols)`: Hitung statistik metrik
  - `_generate_recommendation(analysis, best_model_row, fastest_model_row, repr_metric, time_col)`: Buat rekomendasi berdasarkan analisis

### 7.2 ScenarioAnalyzer (`analysis/scenario_analyzer.py`)
- **Fungsi**: Analisis hasil skenario penelitian
- **Metode Utama**:
  - `analyze_scenario_results(df, backbone_col, condition_col)`: Analisis hasil skenario
  - `_get_metric_columns(df)`: Identifikasi kolom metrik
  - `_get_representative_metric(metric_cols)`: Pilih metrik representatif
  - `_identify_best_scenario(df, repr_metric, backbone_col, condition_col)`: Identifikasi skenario terbaik
  - `_analyze_backbone_performance(df, backbone_col, metric_cols, repr_metric)`: Analisis performa backbone
  - `_analyze_condition_performance(df, condition_col, metric_cols, repr_metric)`: Analisis performa kondisi
  - `_generate_scenario_recommendation(analysis)`: Buat rekomendasi berdasarkan analisis