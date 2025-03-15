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

### CacheManager (manger_cache.py)
- **Fungsi**: Mengelola sistem caching terpusat
- **Metode Utama**:
  - `get(key)`: Ambil data dari cache
  - `put(key, data)`: Simpan data ke cache
  - `cleanup(expired_only)`: Bersihkan cache
  - `get_stats()`: Dapatkan statistik cache

## Domain Dataset 

### DatasetManager (manager.py)
- **Fungsi**: Koordinator alur kerja dataset
- **Metode Utama**:
  - `get_dataloader(split, **kwargs)`: Dapatkan dataloader
  - `download_from_roboflow(workspace, project)`: Download dataset
  - `validate_dataset(split)`: Validasi dataset
  - `augment_dataset(split, **kwargs)`: Augmentasi dataset
  - `balance_dataset(split, strategy)`: Balance dataset

### DatasetLoader (services/loader/dataset_loader.py)
- **Fungsi**: Loading dataset dari storage
- **Metode Utama**:
  - `get_dataloader(split)`: Buat DataLoader PyTorch
  - `_get_split_path(split)`: Path untuk dataset split

### DatasetValidator (services/validator/dataset_validator.py)
- **Fungsi**: Validasi dataset untuk training
- **Metode Utama**:
  - `validate(split)`: Validasi dataset
  - `generate_report()`: Buat laporan validasi

### AugmentationService (services/augmentor/augmentation_service.py)
- **Fungsi**: Layanan augmentasi dataset
- **Metode Utama**:
  - `augment_dataset(split, **kwargs)`: Augmentasi dataset
  - `get_pipeline(types)`: Dapatkan pipeline augmentasi

### MultilayerDataset (components/datasets/multilayer_dataset.py)
- **Fungsi**: Dataset multilayer untuk deteksi
- **Metode Utama**:
  - `__getitem__(idx)`: Ambil item dataset
  - `get_layer_annotation(img_id, layer)`: Anotasi per layer

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