# Rencana Restrukturisasi Model Manager SmartCash

## Tujuan

Restrukturisasi `model_manager.py` dan `backbone_handler.py` untuk mengikuti prinsip Single Responsibility dengan membuat komponen lebih atomic, modular, dan dapat diuji. Fokus implementasi adalah training standar untuk membandingkan dua model dengan ukuran gambar tetap 640x640.

## ⚠️ Peringatan Duplikasi

**HINDARI menduplikasi implementasi yang sudah ada di folder `utils`, `data_manager`, dan `checkpoint_manager`!** Beberapa komponen yang sudah ada:

- `TrainingPipeline` - Gunakan untuk sistem training
- `SmartCashLogger` - Gunakan untuk logging terstruktur 
- `MetricsCalculator` - Gunakan untuk perhitungan metrik
- `CheckpointManager` - Gunakan untuk pengelolaan checkpoint model

## Struktur Folder dan File

```
smartcash/handlers/model/
├── __init__.py                     # Export komponen utama
├── model_manager.py                # Entry point minimal (facade)

├── core/                           # Komponen inti model
│   ├── model_component.py          # Kelas dasar komponen model
│   ├── model_factory.py            # Factory pembuatan model dan arsitektur
│   ├── backbone_factory.py         # Factory pembuatan backbone (dari backbone_handler)
│   ├── optimizer_factory.py        # Factory untuk optimizer dan scheduler
│   ├── model_trainer.py            # Komponen training model
│   └── model_predictor.py          # Komponen prediksi dengan model

├── experiments/                    # Eksperimen dan riset
│   ├── experiment_manager.py       # Manajer eksperimen
│   └── backbone_comparator.py      # Komponen khusus untuk perbandingan backbone

├── observers/                      # Observer untuk monitoring
│   ├── base_observer.py            # Observer dasar
│   ├── metrics_observer.py         # Monitoring metrik training
│   └── colab_observer.py           # Observer khusus Colab

├── integration/                    # Adapter untuk integrasi
│   ├── checkpoint_adapter.py       # Adapter untuk CheckpointManager
│   ├── metrics_adapter.py          # Adapter untuk MetricsCalculator
│   ├── data_adapter.py             # Adapter untuk DataManager
│   └── drive_adapter.py            # Adapter untuk Google Drive

└── visualizations/                 # Visualisasi training dan evaluasi
    ├── metrics_visualizer.py       # Visualisasi metrik training
    └── comparison_visualizer.py    # Visualisasi perbandingan model
```

## Pola Desain yang Digunakan

1. **Facade Pattern**: 
   - `model_manager.py` sebagai entry point dengan antarmuka sederhana
   - Menyembunyikan kompleksitas subsistem factory dan komponen

2. **Factory Pattern**: 
   - Pembuatan model, backbone, optimizer, dan scheduler dengan konfigurasi yang tepat

3. **Adapter Pattern**: 
   - Adapter untuk komponen dari `utils`, `data_manager`, dan `checkpoint_manager`
   - Adapter khusus untuk Google Drive di Colab

4. **Observer Pattern**: 
   - Monitoring metrik dan progress training/evaluasi
   - Observer khusus untuk Colab

## Integrasi dengan Google Colab + Drive

### Deteksi Environment

```python
def is_running_in_colab():
    """Deteksi apakah kode berjalan dalam Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False
```

### DriveAdapter

- `drive_adapter.py` akan menangani integrasi dengan Google Drive
- Mencakup:
  - Mount/unmount Drive secara otomatis
  - Konversi path lokal ke path Drive
  - Handling symlink di Drive
  - Cache status mount Drive

### ColabObserver

- `colab_observer.py` untuk memberikan feedback di lingkungan Colab
- Mendukung:
  - Progress bar yang kompatibel dengan Colab
  - Visualisasi real-time selama training
  - Update grafik metrik secara dinamis

### Modifikasi Path untuk Colab

```python
def get_adjusted_path(path, mount_point="/content/drive"):
    """Sesuaikan path untuk Google Drive jika berjalan di Colab."""
    if is_running_in_colab():
        if str(path).startswith('/'):
            return Path(f"{mount_point}/MyDrive/{str(path).lstrip('/')}")
        return Path(f"{mount_point}/MyDrive/{path}")
    return Path(path)
```

## Kelas Kunci

### ModelManager (Facade)

```python
class ModelManager:
    """
    Manager utama model sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability.
    """
    
    def __init__(self, config=None, logger=None, colab_mode=None):
        # Deteksi otomatis colab jika tidak diberikan
        self.colab_mode = is_running_in_colab() if colab_mode is None else colab_mode
        
        # Setup adapter dan komponen
        # ...
        
    def create_model(self, backbone_type='efficientnet', **kwargs):
        """Buat model baru dengan konfigurasi tertentu."""
        return self.model_factory.create_model(backbone_type, **kwargs)
        
    def train(self, train_loader, val_loader, **kwargs):
        """Train model dengan dataset yang diberikan."""
        return self.model_trainer.train(train_loader, val_loader, **kwargs)
        
    def predict(self, images, **kwargs):
        """Prediksi dengan model yang dimuat."""
        return self.model_predictor.predict(images, **kwargs)
        
    def load_model(self, checkpoint_path, **kwargs):
        """Muat model dari checkpoint."""
        return self.model_factory.load_model(checkpoint_path, **kwargs)
        
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs):
        """Bandingkan beberapa backbone dengan kondisi yang sama."""
        return self.experiment_manager.compare_backbones(backbones, train_loader, val_loader, test_loader, **kwargs)
```

### ModelFactory

```python
class ModelFactory:
    """
    Factory untuk pembuatan model dengan berbagai backbone.
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.backbone_factory = BackboneFactory(config, logger)
        
    def create_model(self, backbone_type='efficientnet', num_classes=None, pretrained=True, **kwargs):
        """
        Buat model dengan backbone tertentu.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            num_classes: Jumlah kelas
            pretrained: Load pretrained weights
            
        Returns:
            Model yang diinisialisasi
        """
        # Ambil jumlah kelas dari config jika tidak diberikan
        if num_classes is None:
            num_classes = self.config.get('model', {}).get('num_classes', 7)
            
        # Buat backbone
        backbone = self.backbone_factory.create_backbone(backbone_type, pretrained)
        
        # Buat model dengan backbone
        model = YOLOv5Model(
            backbone=backbone,
            num_classes=num_classes,
            **kwargs
        )
        
        self.logger.success(f"✅ Model berhasil dibuat dengan backbone {backbone_type}")
        
        return model
    
    def load_model(self, checkpoint_path, **kwargs):
        """
        Muat model dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Model yang dimuat
        """
        # ...
```

### BackboneFactory (dari backbone_handler)

```python
class BackboneFactory:
    """
    Factory untuk pembuatan backbone dengan berbagai arsitektur.
    Implementasi ulang dari BackboneHandler sebagai factory.
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
    def create_backbone(self, backbone_type='efficientnet', pretrained=True, weights_path=None):
        """
        Buat backbone dengan tipe tertentu.
        
        Args:
            backbone_type: Tipe backbone ('efficientnet', 'cspdarknet', dll)
            pretrained: Load pretrained weights
            weights_path: Custom weights path
            
        Returns:
            Backbone yang diinisialisasi
        """
        backbone_type = backbone_type.lower()
        
        if backbone_type == 'efficientnet':
            model = self._create_efficientnet(pretrained)
        elif backbone_type == 'cspdarknet':
            model = self._create_cspdarknet(pretrained)
        else:
            raise ValueError(f"Backbone type {backbone_type} tidak didukung")
            
        # Load weights kustom jika diberikan
        if weights_path:
            self._load_weights(model, weights_path)
            
        self.logger.info(f"🔄 Backbone {backbone_type} berhasil dibuat")
        
        return model
    
    def _create_efficientnet(self, pretrained=True):
        """Buat EfficientNet backbone."""
        # ...
        
    def _create_cspdarknet(self, pretrained=True):
        """Buat CSPDarknet backbone."""
        # ...
        
    def _load_weights(self, model, weights_path):
        """Load weights ke backbone."""
        # ...
```

### ModelTrainer

```python
class ModelTrainer:
    """
    Komponen untuk melatih model dengan pendekatan standar.
    """
    
    def __init__(self, config, logger=None, checkpoint_adapter=None, metrics_adapter=None):
        self.config = config
        self.logger = logger
        self.checkpoint_adapter = checkpoint_adapter
        self.metrics_adapter = metrics_adapter
        self.optimizer_factory = OptimizerFactory(config, logger)
        
    def train(self, train_loader, val_loader, model=None, **kwargs):
        """
        Train model dengan dataset yang diberikan menggunakan pendekatan standar.
        
        Args:
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            model: Model untuk di-train (opsional)
            
        Returns:
            Dict hasil training
        """
        # Buat model jika belum diberikan
        if model is None:
            model_factory = ModelFactory(self.config, self.logger)
            model = model_factory.create_model()
            
        # Setup checkpoint adapter jika belum ada
        if self.checkpoint_adapter is None:
            from smartcash.handlers.model.integration.checkpoint_adapter import CheckpointAdapter
            self.checkpoint_adapter = CheckpointAdapter(self.config, self.logger)
            
        # Setup metrics adapter jika belum ada
        if self.metrics_adapter is None:
            from smartcash.handlers.model.integration.metrics_adapter import MetricsAdapter
            self.metrics_adapter = MetricsAdapter(self.logger)
            
        # Siapkan observer
        observers = kwargs.pop('observers', [])
        
        # Tambahkan ColabObserver jika di Colab
        if is_running_in_colab() and not any(isinstance(obs, ColabObserver) for obs in observers):
            from smartcash.handlers.model.observers.colab_observer import ColabObserver
            observers.append(ColabObserver(self.logger))
            
        # Gunakan TrainingPipeline dari utils
        from smartcash.utils.training import TrainingPipeline
        
        pipeline = TrainingPipeline(
            config=self.config,
            model_handler=model,
            logger=self.logger
        )
        
        # Register callbacks
        for observer in observers:
            pipeline.register_callback('epoch_end', observer.on_epoch_end)
            pipeline.register_callback('training_start', observer.on_training_start)
            pipeline.register_callback('training_end', observer.on_training_end)
        
        # Jalankan training
        results = pipeline.train(
            dataloaders={
                'train': train_loader,
                'val': val_loader
            },
            **kwargs
        )
        
        return results
```

### ExperimentManager

```python
class ExperimentManager:
    """
    Manajer untuk eksperimen model, fokus pada perbandingan backbone.
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.model_factory = ModelFactory(config, logger)
        self.model_trainer = ModelTrainer(config, logger)
        
    def compare_backbones(self, backbones, train_loader, val_loader, test_loader=None, **kwargs):
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone untuk dibandingkan (misalnya ['efficientnet', 'cspdarknet'])
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validation
            test_loader: DataLoader untuk testing (opsional)
            
        Returns:
            Dict hasil perbandingan
        """
        self.logger.info(f"🔬 Membandingkan {len(backbones)} backbone: {', '.join(backbones)}")
        
        results = {}
        
        for backbone_type in backbones:
            self.logger.info(f"🚀 Memulai eksperimen dengan backbone: {backbone_type}")
            
            # Buat model dengan backbone yang ditentukan
            model = self.model_factory.create_model(backbone_type=backbone_type)
            
            # Training model
            training_results = self.model_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                **kwargs
            )
            
            # Evaluasi model dengan test_loader jika disediakan
            eval_results = {}
            if test_loader is not None:
                pass  # Implementasi evaluasi jika diperlukan
                
            # Simpan hasil
            results[backbone_type] = {
                'training': training_results,
                'evaluation': eval_results
            }
            
            self.logger.success(f"✅ Eksperimen backbone {backbone_type} selesai")
        
        # Visualisasi perbandingan hasil
        self._visualize_comparison(results)
        
        return results
    
    def _visualize_comparison(self, results):
        """Visualisasikan perbandingan hasil antar backbone."""
        # Implementasi visualisasi perbandingan
        pass
```

## Integrasi dengan Komponen dari Utils

### Integrasi dengan TrainingPipeline

```python
# Dalam ModelTrainer.train()
from smartcash.utils.training import TrainingPipeline

pipeline = TrainingPipeline(
    config=self.config,
    model_handler=model,
    logger=self.logger
)

# Register callbacks
for observer in observers:
    pipeline.register_callback('epoch_end', observer.on_epoch_end)
    pipeline.register_callback('training_start', observer.on_training_start)
    pipeline.register_callback('training_end', observer.on_training_end)

# Jalankan training
results = pipeline.train(
    dataloaders={
        'train': train_loader,
        'val': val_loader
    },
    resume_from_checkpoint=checkpoint_path,
    save_every=save_every
)
```

### Integrasi dengan CheckpointManager

```python
# Dalam CheckpointAdapter
from smartcash.handlers.checkpoint import CheckpointManager

checkpoint_manager = CheckpointManager(
    output_dir=self.config.get('output_dir', 'runs/train/weights'),
    logger=self.logger
)

# Save checkpoint
checkpoint_result = checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config=self.config,
    epoch=current_epoch,
    metrics=metrics,
    is_best=(val_loss < best_val_loss)
)
```

### Integrasi dengan MetricsCalculator

```python
# Dalam MetricsAdapter
from smartcash.utils.metrics import MetricsCalculator

metrics_calculator = MetricsCalculator()
metrics_calculator.update(predictions, targets)
final_metrics = metrics_calculator.compute()
```

## Menghadapi Keterbatasan Colab

1. **Memori Terbatas**:
   - Support untuk mode `mixed_precision` (FP16)
   - Cleaning cache secara berkala

2. **Keterbatasan GPU**:
   - Deteksi otomatis ketersediaan GPU
   - Fallback ke CPU dengan peringatan performa

3. **Timeout Session**:
   - Sistem checkpoint training otomatis
   - Status penyimpanan otomatis ke Drive

## Contoh Penggunaan Setelah Restrukturisasi

```python
# Import
from smartcash.handlers.model import ModelManager

# Inisialisasi
model_manager = ModelManager(config=config)

# Perbandingan backbone
results = model_manager.compare_backbones(
    backbones=['efficientnet', 'cspdarknet'],
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=30,
    early_stopping=True
)

# Atau training satu model
model = model_manager.create_model(backbone_type='efficientnet')
training_results = model_manager.train(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    epochs=30
)
```