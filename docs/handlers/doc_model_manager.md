# Dokumentasi ModelManager SmartCash

## Deskripsi

`ModelManager` adalah komponen pusat untuk pengelolaan model deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi model.

## Struktur dan Komponen

```
smartcash/handlers/model/
├── __init__.py                     # Export komponen utama
├── model_manager.py                # Entry point utama (facade)
├── model_experiments.py            # Eksperimen model

├── core/                           # Komponen inti model
│   ├── model_component.py          # Kelas dasar komponen model
│   ├── model_factory.py            # Factory pembuatan model
│   ├── backbone_factory.py         # Factory pembuatan backbone
│   ├── optimizer_factory.py        # Factory untuk optimizer
│   ├── model_trainer.py            # Komponen training
│   ├── model_evaluator.py          # Komponen evaluasi
│   └── model_predictor.py          # Komponen prediksi

├── experiments/                    # Eksperimen model
│   ├── experiment_manager.py       # Manajer eksperimen
│   └── backbone_comparator.py      # Perbandingan backbone

├── observers/                      # Observer untuk monitoring
│   ├── model_observer_interface.py # Interface observer
│   ├── metrics_observer.py         # Observer metrik
│   ├── colab_observer.py           # Observer Colab
│   └── experiment_observer.py      # Observer eksperimen

├── integration/                    # Adapter untuk integrasi
│   ├── checkpoint_adapter.py       # Adapter untuk CheckpointManager
│   ├── metrics_adapter.py          # Adapter untuk MetricsCalculator
│   ├── environment_adapter.py      # Adapter untuk environment
│   ├── experiment_adapter.py       # Adapter untuk experiment tracking
│   └── exporter_adapter.py         # Adapter untuk export model
```

## Fitur Utama

### 1. Pembuatan Model

- Dukungan backbone EfficientNet dan CSPDarknet
- Factory pattern untuk pembuatan model yang fleksibel
- Loading model dari checkpoint

### 2. Training Model

- Integrasi dengan TrainingPipeline dari utils
- Monitoring training melalui observer pattern
- Early stopping untuk optimasi training
- Dukungan untuk Google Colab

### 3. Evaluasi Model

- Evaluasi model pada dataset test
- Perhitungan metrik (mAP, precision, recall, F1)
- Pengukuran waktu inferensi

### 4. Prediksi

- Prediksi pada gambar atau batch gambar
- Prediksi pada video dengan visualisasi
- Pre-processing dan post-processing otomatis

### 5. Eksperimen

- Perbandingan backbone dengan kondisi yang sama
- Visualisasi perbandingan performa
- Analisis hasil eksperimen

### 6. Export Model

- Export ke format deployment (TorchScript, ONNX)
- Dukungan untuk half precision

## Kelas Utama

### ModelManager

```python
class ModelManager:
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None
    ):
        """Inisialisasi model manager."""
```

Metode utama:
- `create_model()`: Buat model dengan konfigurasi tertentu
- `load_model()`: Muat model dari checkpoint
- `train()`: Train model dengan dataset
- `evaluate()`: Evaluasi model pada dataset test
- `predict()`: Prediksi dengan model
- `predict_on_video()`: Prediksi pada video
- `compare_backbones()`: Bandingkan backbone
- `export_model()`: Export model untuk deployment

### ModelTrainer

```python
def train(
    self,
    train_loader,
    val_loader,
    model=None,
    checkpoint_path=None,
    observers=None,
    **kwargs
) -> Dict[str, Any]:
    """Train model dengan dataset yang diberikan."""
```

### ModelEvaluator

```python
def evaluate(
    self, 
    test_loader, 
    model=None, 
    checkpoint_path=None, 
    observers=None, 
    **kwargs
) -> Dict:
    """Evaluasi model pada test dataset."""
```

### ModelExperiments

```python
def compare_backbones(
    self,
    backbones: List[str],
    train_loader,
    val_loader,
    test_loader = None,
    parallel: bool = False,
    visualize: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Bandingkan beberapa backbone dengan kondisi yang sama."""
```

## Observer Pattern

`ModelManager` menggunakan observer pattern untuk monitoring:

1. **ModelObserverInterface**: Interface observer untuk model
2. **MetricsObserver**: Monitoring metrik training
3. **ColabObserver**: Visualisasi di Colab
4. **ExperimentObserver**: Monitoring eksperimen

## Format Hasil

### Hasil Training

```python
{
    'epoch': 30,                      # Epoch terakhir
    'best_epoch': 25,                 # Epoch terbaik
    'best_val_loss': 0.125,           # Validation loss terbaik
    'early_stopped': True,            # Flag early stopping
    'best_checkpoint_path': '...',    # Path checkpoint terbaik
    'execution_time': 3600.5          # Waktu eksekusi (detik)
}
```

### Hasil Evaluasi

```python
{
    'mAP': 0.92,                      # Mean Average Precision
    'precision': 0.88,                # Precision
    'recall': 0.89,                   # Recall
    'f1': 0.885,                      # F1 Score
    'execution_time': 120.5,          # Waktu eksekusi (detik)
}
```

### Hasil Prediksi

```python
{
    'num_images': 5,                  # Jumlah gambar
    'detections': [ ... ],            # Hasil deteksi per gambar
    'visualization_paths': [ ... ],   # Path hasil visualisasi
    'execution_time': 0.85,           # Waktu eksekusi (detik)
    'fps': 5.88                       # Frame per detik
}
```

### Hasil Perbandingan Backbone

```python
{
    'experiment_name': '...',         # Nama eksperimen
    'num_backbones': 2,               # Jumlah backbone
    'backbones': ['efficientnet', 'cspdarknet'],  # Backbone
    'results': { ... },               # Hasil per backbone
    'summary': { ... }                # Ringkasan perbandingan
}
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: ModelManager sebagai entry point
2. **Factory Pattern**: Pembuatan komponen model, backbone, dan optimizer
3. **Observer Pattern**: Monitoring training dan evaluasi
4. **Adapter Pattern**: Integrasi dengan komponen lain
5. **Lazy-loading Pattern**: Loading komponen saat dibutuhkan

## Integrasi dengan Google Colab

- **ColabObserver**: Visualisasi real-time di Colab
- **Deteksi Otomatis**: Mendeteksi lingkungan Colab
- **Visualisasi Terintegrasi**: Display metrik di notebook

## Contoh Penggunaan

### Training Model

```python
from smartcash.handlers.model import ModelManager

# Inisialisasi model manager
model_manager = ModelManager(config)

# Buat model dengan backbone EfficientNet
model = model_manager.create_model(backbone_type="efficientnet")

# Train model
results = model_manager.train(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    epochs=30
)
```

### Evaluasi Model

```python
# Evaluasi model
eval_results = model_manager.evaluate(
    test_loader=test_loader,
    model=model
)

print(f"mAP: {eval_results['mAP']:.4f}")
```

### Perbandingan Backbone

```python
# Bandingkan backbone
compare_results = model_manager.compare_backbones(
    backbones=["efficientnet", "cspdarknet"],
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```