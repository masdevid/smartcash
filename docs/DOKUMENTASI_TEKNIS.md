# 📚 Dokumentasi Teknis SmartCash (Updated)

## 📋 Overview

SmartCash adalah sistem deteksi nilai mata uang yang mengintegrasikan YOLOv5 dengan EfficientNet-B4 sebagai backbone melalui timm library. Sistem ini dirancang untuk meningkatkan akurasi deteksi nilai mata uang Rupiah dengan mempertimbangkan berbagai kondisi pengambilan gambar.

## 🔄 Flow Sistem

### 1. Input Processing
- **Input Format**: Gambar RGB dengan preprocessing:
  - Resize ke 640x640 piksel
  - Normalisasi nilai piksel (0-1)
  - Augmentasi data (saat training)

### 2. EfficientNet-B4 Backbone
- **Implementasi**: Menggunakan timm library
  - `features_only=True` untuk mendapatkan intermediate features
  - `out_indices=(2,3,4)` untuk P3, P4, P5 stages
- **Feature Extraction**:
  - P3: Stage 2 features
  - P4: Stage 3 features
  - P5: Stage 4 features
- **Training Control**:
  - Configurable layer freezing
  - Transfer learning dari pretrained weights

### 3. Feature Processing Neck
- **Feature Pyramid Network (FPN)**:
  - Lateral connections untuk channel adaptation
  - Top-down pathway untuk semantic enhancement
  - Multi-scale feature fusion
- **Path Aggregation Network (PAN)**:
  - Bottom-up path augmentation
  - Enhanced feature propagation
  - Adaptive feature pooling

### 4. Detection Head
- **Multi-scale Detection**:
  - Dedicated head untuk setiap skala (P3, P4, P5)
  - Format output: [batch, anchors, height, width, 5+classes]
- **Output Components**:
  - Bounding box coordinates (x, y, w, h)
  - Objectness score
  - Class probabilities (7 denominasi Rupiah)

## 🛠️ Komponen Utama

### EfficientNetAdapter
```python
class EfficientNetAdapter(nn.Module):
    def __init__(self, pretrained=True, trainable_layers=3):
        # Inisialisasi backbone dengan timm
        self.efficientnet = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4)
        )
```

### PreprocessingHandler
```python
class PreprocessingCache:
    """Cache system untuk hasil preprocessing"""
    def __init__(self, cache_dir: str, max_size_gb: float = 1.0):
        # Konfigurasi cache
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
class ImagePreprocessor:
    """Preprocessor untuk dataset uang kertas"""
    def __init__(
        self,
        config_path: str,
        cache_size_gb: float = 1.0
    ):
        # Setup komponen
        self.cache = PreprocessingCache(max_size_gb=cache_size_gb)
        self.coord_normalizer = CoordinateNormalizer()
        self.augmentor = self._setup_augmentations()

class CoordinateNormalizer:
    """Normalizer untuk koordinat label"""
    def normalize_polygon(
        self,
        points: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> List[float]:
        # Normalisasi koordinat ke range [0,1]
        pass
```

### FeatureProcessingNeck
```python
class FeatureProcessingNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.fpn = FeaturePyramidNetwork(in_channels, out_channels)
        self.pan = PathAggregationNetwork(out_channels)
```

### DetectionHead
```python
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=7):
        # Detection layers untuk mata uang
        self.conv = nn.Sequential(...)
```

## 📊 Training Pipeline

### Data Management & Preprocessing

#### Pipeline Preprocessing
1. **Input Data**
   - Dataset split: 70% training, 15% validation, 15% testing
   - Format gambar: JPG/JPEG
   - Format label: TXT (YOLO format & polygon coordinates)

2. **Cache System**
   - Persistent disk cache untuk hasil preprocessing
   - Cache key berdasarkan hash file + parameter
   - LRU cleanup policy dengan configurable max size
   - Tracking cache stats (hit rate, size, file count)

3. **Coordinate Normalization**
   - Normalisasi koordinat polygon ke range [0,1]
   - Backward compatibility dengan format YOLO bbox
   - Validasi dan sanitasi koordinat

4. **Image Processing**
   - Resize ke 640x640 pixels
   - Normalisasi pixel values (mean/std)
   - Augmentasi data:
     - Rotasi dan flipping
     - Variasi pencahayaan
     - Scaling dan cropping

5. **Validasi Output**
   - Verifikasi integritas gambar
   - Validasi format label
   - Statistik preprocessing detail
   - Auto-revalidation

### Optimization
- Transfer learning dari pretrained EfficientNet-B4
- Progressive layer unfreezing
- Learning rate scheduling dengan warmup

### Monitoring
- Logging dengan contextual emojis
- Debug information untuk feature shapes
- Progress tracking dengan tqdm

## 🔍 Evaluasi

### Skenario Pengujian
1. **Baseline**: YOLOv5 dengan CSPDarknet
   - Variasi posisi
   - Variasi pencahayaan
2. **Optimized**: YOLOv5 dengan EfficientNet-B4
   - Variasi posisi
   - Variasi pencahayaan

### Metrik Evaluasi
- Accuracy
- Precision
- Recall
- F1-Score
- mAP
- Inference Time

## 💾 Cache Management

### Struktur Cache
```
.cache/preprocessing/
├── cache_index.json    # Metadata & tracking
└── [hash].pkl         # Cached preprocessing results
```

### Cache Metrics
- Hit Rate: Persentase cache hits vs total requests
- Cache Size: Total ukuran data yang di-cache
- File Count: Jumlah file dalam cache system

### Cleanup Policy
1. **Size-based**:
   - Trigger ketika melebihi max_size_gb
   - Hapus file terlama (LRU)
   - Update index setelah cleanup

2. **Manual**:
   - Method clear_cache() untuk reset
   - Recreate cache structure
   - Reset statistics

## 🚀 Deployment

### Requirements
```text
torch>=1.7.0
timm>=0.6.12
numpy>=1.19.0
opencv-python>=4.5.0
albumentations>=1.0.0
```

### Hardware Requirements
- GPU dengan minimal 6GB VRAM
- CPU dengan 8+ cores untuk preprocessing
- 16GB+ RAM untuk batch processing

### Resource Management
- Memory limit: 60% dari available resources
- Multi-processing untuk data loading
- Batch size optimization

## 📝 Notes
- Backbone menggunakan timm untuk better compatibility
- Channel dimensions diambil otomatis dari EfficientNet
- Debug mode tersedia untuk detailed feature analysis
- Resource usage dibatasi untuk stabilitas sistem