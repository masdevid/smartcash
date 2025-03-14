# DT01 - SmartCash Detection Architecture Refactor Guide

## Struktur Direktori

```
smartcash/detection/
│
├── __init__.py                 # Ekspor komponen publik detection
├── detector.py                 # Koordinator utama untuk deteksi objek
│
├── services/                   # Layanan deteksi spesifik
│   ├── __init__.py
│   │
│   ├── inference/              # Layanan inferensi
│   │   ├── __init__.py
│   │   ├── inference_service.py    # Layanan inferensi utama
│   │   ├── optimizers.py           # Optimasi inferensi
│   │   ├── batch_processor.py      # Pemrosesan batch untuk inferensi
│   │   └── accelerator.py          # Akselerator hardware (CUDA, TensorRT)
│   │
│   ├── postprocessing/         # Layanan pasca-inferensi
│   │   ├── __init__.py
│   │   ├── nms_processor.py        # Non-Maximum Suppression
│   │   ├── confidence_filter.py    # Filter confidence
│   │   ├── bbox_refiner.py         # Perbaikan bounding box
│   │   └── result_formatter.py     # Format hasil deteksi
│   │
│   ├── visualization/          # Layanan visualisasi hasil
│   │   ├── __init__.py
│   │   ├── bbox_visualizer.py      # Visualisasi bounding box
│   │   ├── label_visualizer.py     # Visualisasi label
│   │   ├── confidence_visualizer.py # Visualisasi confidence
│   │   └── heatmap_generator.py    # Generator peta panas (opsional)
│   │
│   └── evaluation/             # Layanan evaluasi model
│       ├── __init__.py
│       ├── metrics_calculator.py   # Penghitung metrik
│       ├── evaluator.py            # Evaluator model
│       ├── benchmark.py            # Benchmark kinerja
│       └── result_analyzer.py      # Analisis hasil deteksi
│
├── models/                     # Model deteksi
│   ├── __init__.py
│   ├── yolov5_model.py             # Implementasi model YOLOv5
│   ├── efficientnet_backbone.py    # Implementasi backbone EfficientNet
│   ├── fpn_pan_neck.py             # Implementasi FPN-PAN neck
│   ├── detection_head.py           # Implementasi detection head
│   └── anchors.py                  # Pengelolaan anchors
│
├── utils/                      # Utilitas deteksi
│   ├── __init__.py
│   ├── preprocess/                 # Utilitas preprocessing
│   │   ├── __init__.py
│   │   ├── image_transform.py      # Transformasi gambar
│   │   ├── normalization.py        # Normalisasi input
│   │   └── augmentation.py         # Augmentasi real-time untuk inferensi
│   │
│   ├── postprocess/                # Utilitas postprocessing
│   │   ├── __init__.py
│   │   ├── bbox_utils.py           # Utilitas bounding box
│   │   ├── nms_utils.py            # Utilitas NMS
│   │   └── score_utils.py          # Utilitas skor/confidence
│   │
│   └── optimization/               # Utilitas optimasi
│       ├── __init__.py
│       ├── weight_quantization.py  # Kuantisasi bobot
│       ├── pruning.py              # Pruning model
│       └── memory_optimization.py  # Optimasi penggunaan memori
│
└── adapters/                  # Adapter untuk integrasi
    ├── __init__.py
    ├── onnx_adapter.py            # Adapter untuk ONNX
    ├── torchscript_adapter.py     # Adapter untuk TorchScript
    ├── tensorrt_adapter.py        # Adapter untuk TensorRT (opsional)
    └── tflite_adapter.py          # Adapter untuk TFLite (opsional)
```

## Konsep Arsitektur

1. **Detector**
   
   File `detector.py` bertindak sebagai titik masuk utama untuk operasi deteksi dalam aplikasi. Menyediakan antarmuka tingkat tinggi untuk layanan deteksi dengan delegasi ke layanan spesifik:
   
   ```python
   class Detector:
       """
       Koordinator utama untuk deteksi objek SmartCash dengan dukungan multi-layer.
       Mengelola proses deteksi end-to-end dengan delegasi ke layanan spesifik.
       """
       
       def __init__(self, model_path, config=None, logger=None):
           """Inisialisasi Detector dengan model dan konfigurasi."""
           self.config = config or {}
           self.logger = logger or get_logger("detector")
           
           # Muat model dan konfigurasi
           self.model = self._load_model(model_path)
           self.layers = self.config.get('layers', ['banknote', 'nominal', 'security'])
           
           # Inisialisasi layanan (lazy loading)
           self._services = {}
           
       def detect(self, image, **kwargs):
           """
           Deteksi objek dalam gambar.
           
           Args:
               image: Gambar input (numpy array atau path)
               **kwargs: Parameter tambahan untuk deteksi
               
           Returns:
               Hasil deteksi dengan bound box dan label
           """
           # Preprocess gambar
           processed_image = self.get_service('preprocessor').process(image)
           
           # Lakukan inferensi
           raw_outputs = self.get_service('inference').infer(processed_image)
           
           # Postprocess hasil
           detections = self.get_service('postprocessor').process(raw_outputs)
           
           return detections
           
       def get_service(self, service_name):
           """Lazy loading untuk service terkait."""
           if service_name not in self._services:
               if service_name == 'inference':
                   from smartcash.detection.services.inference import InferenceService
                   self._services[service_name] = InferenceService(self.model, self.config, self.logger)
               # dan lainnya untuk service-service lain
           return self._services[service_name]
   ```

2. **Layanan Modular**
   
   Setiap layanan memiliki tanggung jawab yang jelas dan terfokus:
   
   ```python
   # services/inference/inference_service.py
   class InferenceService:
       """
       Layanan inferensi untuk model deteksi.
       Mengelola eksekusi model dan optimasi runtime.
       """
       
       def __init__(self, model, config=None, logger=None):
           self.model = model
           self.config = config or {}
           self.logger = logger or get_logger("inference_service")
           
           # Konfigurasi
           self.batch_size = self.config.get('batch_size', 1)
           self.device = self._get_device()
           
           # Optimasi model jika diperlukan
           self.model = self._optimize_model()
           
       def infer(self, image):
           """
           Lakukan inferensi pada gambar yang sudah dipreprocess.
           
           Args:
               image: Gambar preprocessed dalam format tensor
               
           Returns:
               Raw output dari model
           """
           # Pindahkan ke device yang sesuai
           if isinstance(image, np.ndarray):
               image = torch.from_numpy(image).to(self.device)
           else:
               image = image.to(self.device)
               
           # Lakukan inferensi
           with torch.no_grad():
               outputs = self.model(image)
               
           return outputs
   ```

3. **Model Arsitektur**
   
   Komponen model yang modular dan dapat dikonfigurasi:
   
   ```python
   # models/yolov5_model.py
   class SmartCashYOLOv5(nn.Module):
       """
       Model YOLOv5 dengan EfficientNet backbone untuk SmartCash.
       Support deteksi multilayer untuk mata uang Rupiah.
       """
       
       def __init__(self, num_classes_per_layer, img_size=640, backbone='efficientnet-b4'):
           super().__init__()
           
           # Setup layers dan arsitektur
           self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
           self.num_classes_per_layer = num_classes_per_layer
           
           # Inisialisasi backbone
           self.backbone = self._create_backbone(backbone)
           
           # Inisialisasi neck (FPN + PAN)
           self.neck = FPN_PAN_Neck(
               channels=[64, 128, 256, 512], 
               out_channels=[128, 256, 512]
           )
           
           # Inisialisasi detection heads
           self.heads = nn.ModuleDict()
           for layer_name, num_classes in self.num_classes_per_layer.items():
               self.heads[layer_name] = DetectionHead(
                   num_classes=num_classes,
                   anchors=self._get_anchors_for_layer(layer_name)
               )
               
       def forward(self, x):
           """Forward pass."""
           # Backbone
           features = self.backbone(x)
           
           # Neck
           neck_outputs = self.neck(features)
           
           # Detection heads
           results = {}
           for layer_name, head in self.heads.items():
               results[layer_name] = head(neck_outputs)
               
           return results
   ```

4. **Utilitas Terorganisir**
   
   Utilitas diorganisir berdasarkan domain fungsional:
   
   ```python
   # utils/preprocess/image_transform.py
   class ImageTransformer:
       """
       Transformer untuk memproses gambar sebelum inferensi.
       """
       
       def __init__(self, img_size=(640, 640), mean=None, std=None, keep_ratio=True):
           self.img_size = img_size
           self.mean = mean or [0.485, 0.456, 0.406]
           self.std = std or [0.229, 0.224, 0.225]
           self.keep_ratio = keep_ratio
           
       def __call__(self, image):
           """
           Proses gambar untuk inferensi.
           
           Args:
               image: Gambar input (numpy array atau PIL Image)
               
           Returns:
               Tensor gambar yang diproses
           """
           # Resize
           if self.keep_ratio:
               image = self._letterbox(image, new_shape=self.img_size)
           else:
               image = cv2.resize(image, self.img_size)
               
           # Convert to RGB if needed
           if image.shape[2] == 4:  # with alpha channel
               image = image[:, :, :3]
               
           # Normalisasi dan konversi ke tensor
           image = image.astype(np.float32) / 255.0
           image = (image - np.array(self.mean)) / np.array(self.std)
           image = image.transpose(2, 0, 1)  # HWC -> CHW
           image = np.ascontiguousarray(image)
           
           return torch.from_numpy(image).float()
   ```

5. **Adapter untuk Format Model**
   
   Adapter yang memungkinkan penggunaan model dalam berbagai format:
   
   ```python
   # adapters/onnx_adapter.py
   class ONNXModelAdapter:
       """
       Adapter untuk model ONNX untuk memastikan kompatibilitas dengan sistem deteksi.
       """
       
       def __init__(self, model_path, logger=None):
           self.model_path = model_path
           self.logger = logger or get_logger("onnx_adapter")
           self.session = self._load_model()
           
       def _load_model(self):
           """Load model ONNX."""
           import onnxruntime as ort
           providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
           try:
               return ort.InferenceSession(self.model_path, providers=providers)
           except Exception as e:
               self.logger.error(f"❌ Gagal memuat model ONNX: {str(e)}")
               raise
               
       def __call__(self, input_tensor):
           """
           Lakukan inferensi dengan model ONNX.
           
           Args:
               input_tensor: Tensor input
               
           Returns:
               Output dari model dalam format yang kompatibel
           """
           # Convert to numpy if tensor
           if isinstance(input_tensor, torch.Tensor):
               input_numpy = input_tensor.cpu().numpy()
           else:
               input_numpy = input_tensor
               
           # Get input name and run model
           input_name = self.session.get_inputs()[0].name
           outputs = self.session.run(None, {input_name: input_numpy})
           
           # Post-process to match PyTorch model output format
           return self._format_output(outputs)
   ```

## Implementasi Detektor Multilayer

Arsitektur deteksi dirancang untuk mendukung deteksi multilayer untuk mata uang Rupiah, dengan setiap layer mendeteksi aspek berbeda dari mata uang:

1. **Layer Banknote**: Deteksi uang kertas secara keseluruhan
2. **Layer Nominal**: Deteksi area dengan nilai nominal
3. **Layer Security**: Deteksi fitur keamanan

Contoh implementasi:

```python
# detector.py
def detect_multilayer(self, image, threshold=0.25):
    """
    Deteksi multilayer pada gambar.
    
    Args:
        image: Gambar input
        threshold: Threshold confidence untuk deteksi
        
    Returns:
        Dict hasil deteksi per layer
    """
    # Preprocess dan inferensi
    processed_image = self.get_service('preprocessor').process(image)
    raw_outputs = self.get_service('inference').infer(processed_image)
    
    # Postprocess untuk setiap layer
    results = {}
    for layer in self.layers:
        if layer in raw_outputs:
            layer_threshold = self.config.get(f'{layer}_threshold', threshold)
            results[layer] = self.get_service('postprocessor').process_layer(
                raw_outputs[layer], layer=layer, threshold=layer_threshold
            )
    
    return results
```

## Optimasi Kinerja

Arsitektur ini menyediakan beberapa strategi optimasi:

1. **Batch Processing**: Memproses multiple gambar sekaligus
2. **Kuantisasi Model**: Mengurangi presisi model untuk inference lebih cepat
3. **Hardware Acceleration**: Support untuk GPU, TensorRT dan format optimasi lainnya

Contoh implementasi batch processing:

```python
# services/inference/batch_processor.py
class BatchProcessor:
    """
    Processor untuk batch images untuk optimasi inferensi.
    """
    
    def __init__(self, model, batch_size=8, logger=None):
        self.model = model
        self.batch_size = batch_size
        self.logger = logger or get_logger("batch_processor")
        
    def process(self, images):
        """
        Proses batch gambar untuk inferensi efisien.
        
        Args:
            images: List gambar input
            
        Returns:
            List hasil deteksi
        """
        # Bagi gambar menjadi batches
        num_images = len(images)
        batches = [images[i:i+self.batch_size] for i in range(0, num_images, self.batch_size)]
        
        results = []
        for batch in tqdm(batches, desc="🔄 Memproses batch"):
            # Proses batch
            batch_tensor = torch.stack(batch)
            batch_outputs = self.model(batch_tensor)
            
            # Gabungkan hasil
            results.extend(batch_outputs)
            
        return results
```

## Evaluasi Model

Sistem evaluasi komprehensif untuk pengukuran dan benchmark kinerja:

```python
# services/evaluation/evaluator.py
class DetectionEvaluator:
    """
    Evaluator untuk model deteksi objek.
    Menghitung metrik seperti mAP, precision, recall.
    """
    
    def __init__(self, model, data_loader, config=None, logger=None):
        self.model = model
        self.data_loader = data_loader
        self.config = config or {}
        self.logger = logger or get_logger("detection_evaluator")
        
        # Inisialisasi penghitung metrik
        self.metrics_calculator = MetricsCalculator()
        
    def evaluate(self, iou_threshold=0.5, conf_threshold=0.25):
        """
        Evaluasi model pada dataset.
        
        Args:
            iou_threshold: Threshold untuk IoU matching
            conf_threshold: Threshold untuk confidence
            
        Returns:
            Dict berisi hasil evaluasi
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        # Progress bar
        progress = tqdm(self.data_loader, desc="🔍 Evaluasi model")
        
        # Evaluasi pada setiap batch
        for batch in progress:
            images, targets = batch
            
            # Inferensi
            with torch.no_grad():
                predictions = self.model(images)
            
            # Update metrik
            self.metrics_calculator.update(predictions, targets)
            
        # Compute final metrics
        return self.metrics_calculator.compute()
```

## Visualisasi Hasil

Komponen visualisasi untuk menampilkan hasil deteksi:

```python
# services/visualization/bbox_visualizer.py
class BoundingBoxVisualizer:
    """
    Visualizer untuk hasil deteksi dengan bounding box.
    """
    
    def __init__(self, class_names, colors=None, line_thickness=2):
        self.class_names = class_names
        self.colors = colors or self._generate_colors(len(class_names))
        self.line_thickness = line_thickness
        
    def draw_detections(self, image, detections, draw_labels=True, draw_confidence=True):
        """
        Gambar deteksi pada gambar.
        
        Args:
            image: Gambar original
            detections: Hasil deteksi
            draw_labels: Flag untuk menggambar label
            draw_confidence: Flag untuk menggambar confidence
            
        Returns:
            Gambar dengan deteksi
        """
        # Copy gambar untuk tidak memodifikasi original
        img = image.copy()
        
        # Gambar setiap deteksi
        for det in detections:
            # Extract info
            box = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            
            # Dapatkan warna dan nama kelas
            color = self.colors[cls_id % len(self.colors)]
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class-{cls_id}"
            
            # Gambar box
            self._draw_box(img, box, color)
            
            # Gambar label dan confidence jika diminta
            if draw_labels or draw_confidence:
                label = ""
                if draw_labels:
                    label += class_name
                if draw_confidence:
                    label += f" {conf:.2f}" if label else f"{conf:.2f}"
                    
                self._draw_label(img, box, label, color)
                
        return img
```

## Panduan Penggunaan

### Penggunaan Dasar

```python
from smartcash.detection.detector import Detector

# Inisialisasi detector
detector = Detector(model_path="models/smartcash_yolov5.pt")

# Deteksi pada gambar
image = cv2.imread("sample.jpg")
results = detector.detect(image)

# Proses hasil
for layer, detections in results.items():
    print(f"Layer {layer}: {len(detections)} deteksi")
    for det in detections:
        print(f"  Class {det['class_id']} ({det['class_name']}): {det['confidence']:.2f}")
```

### Visualisasi Hasil

```python
from smartcash.detection.services.visualization import BoundingBoxVisualizer

# Buat visualizer dengan nama kelas
class_names = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
visualizer = BoundingBoxVisualizer(class_names)

# Visualisasi hasil deteksi
results = detector.detect(image)
vis_image = visualizer.draw_detections(image, results["banknote"])

# Tampilkan atau simpan gambar
cv2.imshow("Deteksi", vis_image)
cv2.waitKey(0)
# atau
cv2.imwrite("hasil_deteksi.jpg", vis_image)
```

### Evaluasi Model

```python
from smartcash.detection.services.evaluation import DetectionEvaluator

# Buat evaluator
evaluator = DetectionEvaluator(detector.model, val_loader)

# Evaluasi model
metrics = evaluator.evaluate()

# Print hasil
print(f"mAP: {metrics['mAP']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

### Konversi Model

```python
from smartcash.detection.utils.optimization import ModelExporter

# Inisialisasi exporter
exporter = ModelExporter(detector.model)

# Export ke ONNX
onnx_path = exporter.export_to_onnx("models/smartcash_model.onnx")

# Export ke TorchScript
torchscript_path = exporter.export_to_torchscript("models/smartcash_model.pt")
```

## Roadmap Pengembangan

Untuk pengembangan masa depan, beberapa area yang dipertimbangkan:

1. **Dynamic Quantization**: Implementasi kuantisasi dinamis untuk peningkatan performa pada perangkat edge
2. **Mobile Deployment**: Optimasi untuk deployment ke perangkat mobile 
3. **Advanced Detection Features**: 
   - Object Tracking
   - Instance Segmentation
   - Deteksi Keaslian Uang
4. **Integration with Other Services**:
   - API integrations
   - Cloud deployment options
   - Container-based deployment

## Kesimpulan

Arsitektur detection SmartCash dirancang untuk memberikan:

1. **Modularitas**: Komponen terpisah dengan tanggung jawab jelas
2. **Fleksibilitas**: Support untuk berbagai format model dan hardware
3. **Kinerja Tinggi**: Strategi optimasi yang terintegrasi
4. **Skalabilitas**: Mudah dikembangkan dan diintegrasikan

Dengan struktur ini, sistem dapat mendukung kasus penggunaan dari penelitian hingga produksi dan dapat beradaptasi dengan kebutuhan baru di masa depan.
