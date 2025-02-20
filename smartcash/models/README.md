# Dokumentasi Perubahan Arsitektur Model SmartCash

## 1. Perubahan Struktur File

### File yang Dihapus
- `models/yolov5_efficient.py` - Digantikan oleh implementasi yang lebih modular
- `models/backbones/efficient_adapter.py` - Dipindahkan ke sistem backbone baru
- `models/base_trainer.py` - Training logic dipindahkan ke sistem eksperimen
- `models/yolo_trainer.py` - Digantikan oleh BackboneExperiment

### File Baru
```
models/
├── backbones/
│   ├── base.py           # Interface standar untuk backbone
│   ├── cspdarknet.py     # Implementasi CSPDarknet
│   └── efficientnet.py   # Implementasi EfficientNet yang baru
├── detection_head.py      # Detection head dengan opsi multi-layer
├── losses.py             # Implementasi loss function
├── yolov5_model.py       # Model utama yang baru
└── experiments/
    └── backbone_experiment.py  # Template eksperimen backbone
```

## 2. Perubahan Arsitektur

### Sebelum
- Implementasi monolitik dengan EfficientNet backbone
- Kurang fleksibel untuk eksperimen
- Training logic tersebar di beberapa file
- Tidak ada standarisasi interface backbone

### Sesudah
- Arsitektur modular dengan backbone yang bisa diganti
- Interface backbone yang terstandarisasi
- Support untuk deteksi multi-layer (opsional)
- Training logic terpusat di sistem eksperimen

## 3. Fitur Utama yang Ditambahkan

### BaseBackbone Interface
```python
class BaseBackbone(ABC, nn.Module):
    @abstractmethod
    def get_output_channels(self) -> List[int]
    
    @abstractmethod
    def get_output_shapes(self) -> List[Tuple[int, int]]
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]
```

### Sistem Multi-layer Detection
- Support untuk deteksi berlapis:
  - Layer Banknote: Deteksi uang kertas utuh
  - Layer Nominal: Deteksi area nominal
  - Layer Security: Deteksi fitur keamanan
- Bisa diaktifkan/nonaktifkan sesuai kebutuhan

### Loss Calculation
- Support single dan multi-layer detection
- Komponen loss terpisah per layer
- Weight yang bisa dikustomisasi

## 4. Cara Penggunaan

### Single Layer Detection (Default)
```python
model = YOLOv5(
    num_classes=7,
    backbone_type="efficientnet"
)
```

### Multi Layer Detection
```python
model = YOLOv5(
    backbone_type="efficientnet",
    layers=['banknote', 'nominal', 'security']
)
```

### Eksperimen Backbone
```python
experiment = BackboneExperiment(
    config_path="configs/base_config.yaml",
    experiment_name="backbone_comparison"
)

# Jalankan eksperimen
experiment.run_experiment_scenario("position", "cspdarknet")
experiment.run_experiment_scenario("position", "efficientnet")
```

## 5. Keuntungan Perubahan

1. **Modularitas**
   - Komponen yang bisa diganti-ganti
   - Eksperimen lebih mudah dilakukan
   - Kode lebih mudah dimaintain

2. **Fleksibilitas**
   - Support single/multi layer detection
   - Backbone bisa diganti dengan mudah
   - Eksperimen bisa dikustomisasi

3. **Standarisasi**
   - Interface backbone yang konsisten
   - Format input/output yang terstandar
   - Struktur eksperimen yang seragam

4. **Ekstensibilitas**
   - Mudah menambah backbone baru 
   - Mudah menambah layer deteksi
   - Mudah mengembangkan fitur baru

## 6. Catatan Implementasi

### TODO
1. Implementasi detail target matching di loss function
2. Optimasi kalkulasi IoU/GIoU
3. Implementasi evaluasi per layer untuk multi-layer detection

### Keterbatasan
1. Multi-layer detection membutuhkan lebih banyak memori
2. Waktu training lebih lama untuk multi-layer
3. Perlu dataset dengan anotasi berlapis