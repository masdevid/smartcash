# SmartCash v2 - Domain Detection

## 1. File Structure
```
smartcash/detection/
├── init.py                 # Ekspor komponen deteksi dengan all
├── detector.py                 # Detector: Koordinator utama proses deteksi
├── services/
│   ├── init.py             # Ekspor layanan detection dengan all
│   ├── inference/              # Layanan inferensi
│   │   ├── init.py         # Ekspor komponen inferensi dengan all
│   │   ├── inference_service.py # InferenceService: Koordinator inferensi model
│   │   ├── accelerator.py      # HardwareAccelerator: Abstraksi hardware (CPU/GPU/TPU)
│   │   ├── batch_processor.py  # BatchProcessor: Processor batch gambar paralel
│   │   └── optimizers.py       # ModelOptimizer: Optimasi model (ONNX, TorchScript)
│   ├── postprocessing/         # Layanan pasca-inferensi
│   │   ├── init.py         # Ekspor komponen postprocessing dengan all
│   │   ├── postprocessing_service.py # PostprocessingService: Koordinator postprocessing
│   │   ├── confidence_filter.py # ConfidenceFilter: Filter confidence dengan threshold
│   │   ├── bbox_refiner.py     # BBoxRefiner: Perbaikan bounding box
│   │   └── result_formatter.py # ResultFormatter: Format hasil (JSON, CSV, YOLO, COCO)
│   └── visualization_adapter.py # Adapter visualisasi dari domain model
├── handlers/                   # Handler untuk berbagai skenario deteksi
│   ├── init.py             # Ekspor handler dengan all
│   ├── detection_handler.py    # DetectionHandler: Handler untuk deteksi gambar tunggal
│   ├── batch_handler.py        # BatchHandler: Handler untuk deteksi batch (folder/zip)
│   ├── video_handler.py        # VideoHandler: Handler untuk deteksi video dan webcam
│   └── integration_handler.py  # IntegrationHandler: Integrasi dengan UI/API (async)
└── adapters/
├── init.py             # Ekspor adapter dengan all
├── onnx_adapter.py         # ONNXModelAdapter: Adapter untuk model ONNX
└── torchscript_adapter.py  # TorchScriptAdapter: Adapter untuk model TorchScript
```

## 2. Class and Methods Mapping

## 2. Class and Methods Mapping

### Detector (detector.py)
- **Fungsi**: Koordinator utama proses deteksi
- **Metode Utama**:
  - `__init__(model_manager, prediction_service, inference_service, postprocessing_service, visualization_adapter, logger)`: Inisialisasi detector dengan dependensi
  - `detect(image, conf_threshold, iou_threshold, with_visualization)`: Deteksi pada gambar tunggal dengan opsi visualisasi
  - `detect_multilayer(image, threshold)`: Deteksi multilayer dengan threshold khusus per layer
  - `detect_batch(images, conf_threshold, iou_threshold)`: Deteksi pada batch gambar
  - `visualize(image, detections, conf_threshold, show_labels, show_conf, filename)`: Visualisasi hasil deteksi
  - `get_model_info()`: Dapatkan informasi model yang digunakan
  - `set_active_layer(layer_name)`: Set layer aktif untuk deteksi
  - `warm_up(input_shape)`: Warm-up model sebelum inferensi

### DetectionVisualizationAdapter (services/visualization_adapter.py)
- **Fungsi**: Adapter visualisasi dari domain model ke domain detection
- **Metode Utama**:
  - `__init__(detection_output_dir, metrics_output_dir, logger)`: Inisialisasi adapter visualisasi
  - `visualize_detection(image, detections, filename, conf_threshold, show_labels, show_conf)`: Visualisasi hasil deteksi
  - `plot_confusion_matrix(cm, class_names, title, filename, normalize)`: Visualisasi confusion matrix
  - `visualize_model_comparison(comparison_data, metric_cols, title, filename)`: Visualisasi perbandingan model
  - `save_visualization(image, detections, output_path)`: Simpan hasil visualisasi
  - `create_detection_summary(detections, output_path)`: Buat ringkasan deteksi dalam format visual

### InferenceService (services/inference/inference_service.py)
- **Fungsi**: Koordinator layanan inferensi model
- **Metode Utama**:
  - `__init__(prediction_service, postprocessing_service, accelerator, logger)`: Inisialisasi layanan dengan dependensi
  - `infer(image, conf_threshold, iou_threshold)`: Inferensi pada gambar tunggal
  - `batch_infer(images, conf_threshold, iou_threshold)`: Inferensi pada batch gambar
  - `visualize(image, detections)`: Visualisasi hasil inferensi
  - `optimize_model(target_format, **kwargs)`: Optimasi model untuk inferensi
  - `get_model_info()`: Informasi model yang digunakan
  - `preprocess_input(image)`: Preproses input sebelum inferensi
  - `set_batch_size(batch_size)`: Set ukuran batch untuk inferensi

### HardwareAccelerator (services/inference/accelerator.py)
- **Fungsi**: Abstraksi hardware untuk akselerasi inferensi
- **Metode Utama**:
  - `__init__(accelerator_type, device_id, use_fp16, logger)`: Inisialisasi akselerator
  - `setup()`: Setup akselerator untuk inferensi
  - `get_device()`: Dapatkan device yang dikonfigurasi
  - `get_device_info()`: Informasi device (CUDA, MPS, CPU, etc)
  - `_auto_detect_hardware()`: Deteksi otomatis hardware terbaik
  - `_setup_cuda()`: Setup untuk CUDA
  - `_setup_mps()`: Setup untuk Apple MPS
  - `_setup_tpu()`: Setup untuk Google TPU
  - `_setup_rocm()`: Setup untuk AMD ROCm
  - `_setup_cpu()`: Setup untuk CPU
  - `supports_fp16()`: Cek dukungan FP16
  - `cleanup()`: Bersihkan resource akselerator

### BatchProcessor (services/inference/batch_processor.py)
- **Fungsi**: Processor untuk inferensi batch gambar paralel
- **Metode Utama**:
  - `__init__(inference_service, output_dir, num_workers, batch_size, logger)`: Inisialisasi processor
  - `process_directory(input_dir, output_dir, extensions, recursive, conf_threshold, iou_threshold, save_results, save_visualizations, result_format, callback)`: Proses semua gambar dalam direktori
  - `process_batch(images, output_dir, filenames, conf_threshold, iou_threshold, save_results, save_visualizations, result_format)`: Proses batch gambar yang sudah dimuat
  - `_save_result(img_path, detections, output_path, save_visualization, format)`: Simpan hasil deteksi
  - `_parallel_process(inputs, callback)`: Proses paralel dengan thread/process pool
  - `get_progress()`: Dapatkan status progres pemrosesan batch

### ModelOptimizer (services/inference/optimizers.py)
- **Fungsi**: Utilitas optimasi model untuk inferensi
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi optimizer
  - `optimize_to_onnx(model, output_path, input_shape, dynamic_axes, opset_version, simplify)`: Optimasi ke ONNX
  - `optimize_to_torchscript(model, output_path, input_shape, method)`: Optimasi ke TorchScript
  - `optimize_to_tensorrt(onnx_path, output_path, fp16_mode, int8_mode, workspace_size)`: Optimasi ke TensorRT
  - `optimize_to_tflite(model_path, output_path, quantize, input_shape)`: Optimasi ke TFLite
  - `optimize_model(model, model_format, output_path, **kwargs)`: Optimasi model ke format yang ditentukan
  - `verify_optimization(original_model, optimized_model, input_data)`: Verifikasi hasil optimasi
  - `get_optimization_stats()`: Dapatkan statistik optimasi (ukuran, waktu)

### PostprocessingService (services/postprocessing/postprocessing_service.py)
- **Fungsi**: Koordinator postprocessing hasil deteksi
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi service
  - `process(detections, conf_threshold, iou_threshold, refine_boxes, class_specific_nms, max_detections)`: Proses postprocessing lengkap
  - `apply_nms(detections, iou_threshold, class_specific)`: Terapkan Non-Max Suppression
  - `filter_by_confidence(detections, conf_threshold)`: Filter berdasarkan confidence
  - `refine_bboxes(detections, image_dims)`: Perbaiki bounding box
  - `limit_detections(detections, max_detections)`: Batasi jumlah deteksi
  - `get_stats()`: Dapatkan statistik postprocessing

### ConfidenceFilter (services/postprocessing/confidence_filter.py)
- **Fungsi**: Filter deteksi berdasarkan confidence threshold
- **Metode Utama**:
  - `__init__(default_threshold, class_thresholds, logger)`: Inisialisasi filter
  - `process(detections, global_threshold)`: Filter deteksi berdasarkan threshold
  - `set_threshold(class_id, threshold)`: Set threshold per kelas
  - `get_threshold(class_id)`: Dapatkan threshold untuk kelas
  - `reset_thresholds()`: Reset semua threshold ke default
  - `apply_layer_thresholds(detections, layer_config)`: Terapkan threshold per layer

### BBoxRefiner (services/postprocessing/bbox_refiner.py)
- **Fungsi**: Perbaikan bounding box hasil deteksi
- **Metode Utama**:
  - `__init__(clip_boxes, expand_factor, logger)`: Inisialisasi refiner
  - `process(detections, image_width, image_height, specific_classes)`: Perbaiki bounding box deteksi
  - `_expand_bbox(bbox, factor)`: Ekspansi bbox dengan factor tertentu
  - `_clip_bbox(bbox)`: Clip bbox ke range [0,1]
  - `_fix_absolute_bbox(bbox, img_width, img_height)`: Perbaiki bbox dalam koordinat absolut
  - `_merge_overlapping_bboxes(bboxes, iou_threshold)`: Gabungkan bbox yang tumpang tindih

### ResultFormatter (services/postprocessing/result_formatter.py)
- **Fungsi**: Format hasil deteksi ke berbagai format
- **Metode Utama**:
  - `__init__(logger)`: Inisialisasi formatter
  - `to_json(detections, include_metadata, pretty)`: Format ke JSON
  - `to_csv(detections, include_header)`: Format ke CSV
  - `to_yolo_format(detections)`: Format ke format YOLO
  - `to_coco_format(detections, image_id, image_width, image_height)`: Format ke COCO
  - `format_detections(detections, format, **kwargs)`: Format dengan format yang ditentukan
  - `to_dict(detections)`: Konversi ke dictionary Python
  - `add_metadata(detections, metadata)`: Tambahkan metadata ke hasil

### DetectionHandler (handlers/detection_handler.py)
- **Fungsi**: Handler untuk deteksi gambar tunggal
- **Metode Utama**:
  - `__init__(inference_service, postprocessing_service, logger)`: Inisialisasi handler
  - `detect(image, conf_threshold, iou_threshold, apply_postprocessing, return_visualization)`: Deteksi objek pada gambar
  - `save_result(detections, output_path, image, save_visualization, format)`: Simpan hasil deteksi ke file
  - `preprocess_image(image)`: Preproses gambar sebelum deteksi
  - `validate_input(image)`: Validasi input gambar

### BatchHandler (handlers/batch_handler.py)
- **Fungsi**: Handler untuk deteksi batch/kumpulan gambar
- **Metode Utama**:
  - `__init__(detection_handler, num_workers, batch_size, max_batch_size, logger)`: Inisialisasi handler batch
  - `detect_directory(input_dir, output_dir, extensions, recursive, conf_threshold, iou_threshold, save_results, save_visualizations, result_format)`: Deteksi pada semua gambar di direktori
  - `detect_zip(zip_path, output_dir, conf_threshold, iou_threshold, extensions, save_extracted)`: Deteksi pada gambar dalam file ZIP
  - `process_batch(images, filenames, output_dir)`: Proses batch gambar
  - `validate_directory(input_dir)`: Validasi direktori masukan
  - `get_batch_stats()`: Dapatkan statistik pemrosesan batch

### VideoHandler (handlers/video_handler.py)
- **Fungsi**: Handler untuk deteksi video dan webcam
- **Metode Utama**:
  - `__init__(detection_handler, logger)`: Inisialisasi handler
  - `detect_video(video_path, output_path, conf_threshold, iou_threshold, start_frame, end_frame, step, show_progress, show_preview, overlay_info, callback)`: Deteksi pada file video
  - `detect_webcam(camera_id, output_path, conf_threshold, iou_threshold, max_frames, show_preview, overlay_info, callback)`: Deteksi pada webcam
  - `_process_frame(frame, conf_threshold, iou_threshold)`: Proses satu frame
  - `_save_video(frames, detections, output_path, fps)`: Simpan video dengan hasil deteksi
  - `_overlay_detections(frame, detections)`: Overlay deteksi pada frame
  - `stop_stream()`: Hentikan streaming webcam

### IntegrationHandler (handlers/integration_handler.py)
- **Fungsi**: Handler untuk integrasi dengan UI/API secara asinkron
- **Metode Utama**:
  - `__init__(detection_handler, logger)`: Inisialisasi handler
  - `async_detect(image, conf_threshold, iou_threshold)`: Deteksi asinkron pada gambar
  - `async_batch_detect(images, conf_threshold, iou_threshold)`: Deteksi asinkron pada batch
  - `async_video_detect(video_path, conf_threshold, iou_threshold, callback)`: Deteksi asinkron pada video
  - `queue_detection_task(input_data, task_type)`: Tambahkan tugas ke antrian
  - `get_task_status(task_id)`: Dapatkan status tugas
  - `cancel_task(task_id)`: Batalkan tugas tertentu

### ONNXModelAdapter (adapters/onnx_adapter.py)
- **Fungsi**: Adapter untuk model ONNX
- **Metode Utama**:
  - `__init__(model_path, logger)`: Inisialisasi adapter
  - `load_model()`: Muat model ONNX
  - `infer(inputs)`: Lakukan inferensi dengan ONNX runtime
  - `get_input_details()`: Dapatkan detail input model
  - `get_output_details()`: Dapatkan detail output model
  - `optimize(onnx_path, output_path)`: Optimasi model ONNX

### TorchScriptAdapter (adapters/torchscript_adapter.py)
- **Fungsi**: Adapter untuk model TorchScript
- **Metode Utama**:
  - `__init__(model_path, logger)`: Inisialisasi adapter
  - `load_model()`: Muat model TorchScript
  - `infer(inputs)`: Lakukan inferensi dengan TorchScript
  - `to_device(device)`: Pindahkan model ke device tertentu
  - `get_model_info()`: Dapatkan informasi model