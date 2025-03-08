# Ringkasan Perbaikan Implementasi Training

## Perubahan Utama Training Pipeline

Sistem training pipeline telah direstrukturisasi secara signifikan dari pendekatan berbasis fungsi tunggal menjadi **paket modular** dengan kelas-kelas khusus untuk berbagai aspek training. Berikut ringkasan perubahan utama:

### 1. Reorganisasi Struktur

**Sebelum**: Semua logika training berada dalam satu file `training_pipeline.py`
**Sesudah**: Paket terstruktur dengan komponen khusus

```
utils/training/
├── __init__.py           # Ekspor komponen utama
├── training_pipeline.py  # Kelas utama pipeline training
├── training_callbacks.py # Sistem callback untuk event handling
├── training_metrics.py   # Pengelolaan metrik training
├── training_epoch.py     # Handler untuk satu epoch training
└── validation_epoch.py   # Handler untuk satu epoch validasi
```

### 2. Pendekatan Berorientasi Objek

**Sebelum**: Fungsi tunggal `train_model()` tanpa state
**Sesudah**: Kelas-kelas dengan state dan metode:

- `TrainingPipeline` - Kelas utama yang mengkoordinasikan proses training
- `TrainingCallbacks` - Sistem event-based callbacks
- `TrainingMetrics` - Pengelolaan history metrik dan persistensi
- `TrainingEpoch` - Handler untuk satu epoch training
- `ValidationEpoch` - Handler untuk satu epoch validasi

### 3. Kemampuan Konfigurasi yang Ditingkatkan

**Sebelum**: Parameter fungsi dengan nilai default sederhana
**Sesudah**: Konfigurasi yang kaya melalui:

- Dictionary konfigurasi terstruktur
- State objek (direktori output, logger, dll)
- Opsi persistensi yang dapat dikonfigurasi
- Integrasi dengan berbagai model dan dataset

### 4. Pembagian Concern yang Lebih Baik

Training flow sekarang dipisahkan berdasarkan tanggung jawab:

- **Pipeline**: Koordinasi keseluruhan proses
- **Callbacks**: Komunikasi dengan komponen eksternal pada tiap event
- **Metrics**: Tracking dan persistensi metrik
- **Training Epoch**: Eksekusi satu epoch training
- **Validation Epoch**: Evaluasi model pada data validasi

### 5. Fitur Baru Utama

#### Sistem Callback

- Event-driven architecture untuk hook di berbagai titik training
- Support untuk callbacks kustom tanpa perlu modifikasi pipeline inti
- Implementasi standar untuk checkpoint saving, monitoring, dll
- Callback yang dapat diprioritaskan

#### Pengelolaan Metrik yang Ditingkatkan

- Tracking history metrik yang lengkap
- Auto-logging ke CSV, JSON, dan TensorBoard
- Perhitungan metrik lanjutan (per-kelas, mAP, confusion matrix)
- Visualisasi metrik terintegrasi

#### Resume Training

- Kemampuan untuk melanjutkan training dari checkpoint
- Integrasi state optimizer dan learning rate scheduler
- Restore history metrik lengkap
- Recovery dari interupsi

#### Managed Early Stopping

- Konfigurasi multi-metrik untuk early stopping
- Early stopping berdasarkan custom metric
- Callback notification saat early stop
- Restore model terbaik secara otomatis

## Contoh Transformasi Kode

### Contoh 1: Inisialisasi Training

**Kode Lama**:
```python
from smartcash.utils.training_pipeline import train_model

# Parameter training
num_epochs = 30
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training model
results = train_model(
    model, train_loader, val_loader, 
    num_epochs=num_epochs, 
    lr=learning_rate, 
    device=device
)
```

**Kode Baru**:
```python
from smartcash.utils.training import TrainingPipeline
from smartcash.models import ModelHandler
from smartcash.data import DataManager

# Konfigurasi training
config = {
    'training': {
        'epochs': 30,
        'batch_size': 16,
        'early_stopping_patience': 10
    },
    'optimization': {
        'lr': 0.001,
        'weight_decay': 1e-5,
        'scheduler': 'cosine'
    },
    'output_dir': 'runs/train'
}

# Inisialisasi handler
model_handler = ModelHandler(config)
data_manager = DataManager(config)

# Inisialisasi pipeline
pipeline = TrainingPipeline(
    config=config,
    model_handler=model_handler,
    data_manager=data_manager,
    logger=logger
)

# Jalankan training
results = pipeline.train()
```

### Contoh 2: Kustomisasi Training dengan Callbacks

**Kode Lama**:
```python
# Tidak ada dukungan callback dalam versi lama
# Diperlukan modifikasi langsung pada train_model()

# Modifikasi 
def train_model_custom(model, train_loader, val_loader, **kwargs):
    # Copy entire train_model code and modify
    # ...
    # Custom logging
    print(f"Epoch {epoch}, val_loss: {val_loss:.4f}")
    # ...
    # Custom checkpoint saving
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
    # ...
```

**Kode Baru**:
```python
from smartcash.utils.training import TrainingPipeline

# Inisialisasi pipeline
pipeline = TrainingPipeline(config=config, model_handler=model_handler)

# Callback untuk logging metrik kustom
def log_metrics(epoch, metrics, **kwargs):
    logger.info(f"Epoch {epoch}: val_loss={metrics['val_loss']:.4f}")
    
    # Ekspor metrik ke file CSV kustom
    with open('custom_metrics.csv', 'a') as f:
        f.write(f"{epoch},{metrics['train_loss']},{metrics['val_loss']}\n")

# Callback untuk checkpoint kustom
def on_checkpoint_saved(epoch, checkpoint_info, is_best, **kwargs):
    logger.success(f"Checkpoint disimpan untuk epoch {epoch}")
    
    # Lakukan backup tambahan jika model terbaik
    if is_best:
        import shutil
        shutil.copy(checkpoint_info['path'], 'backup/best_model.pt')

# Register callbacks
pipeline.register_callback('epoch_end', log_metrics)
pipeline.register_callback('checkpoint_saved', on_checkpoint_saved)

# Jalankan training dengan callbacks terdaftar
results = pipeline.train()
```

### Contoh 3: Fitur Baru - Resume Training

```python
from smartcash.utils.training import TrainingPipeline

# Inisialisasi pipeline
pipeline = TrainingPipeline(config=config, model_handler=model_handler)

# Lanjutkan training dari checkpoint
results = pipeline.train(
    resume_from_checkpoint='path/to/checkpoint.pt',
    save_every=5
)

# Inspeksi hasil training
print(f"Training dilanjutkan dari epoch {results['start_epoch']}")
print(f"Total epoch: {results['epochs_trained']}")
print(f"Best val_loss: {results['best_val_loss']:.4f}")
```

## Pembaruan Output Training

### 1. Metrik Training yang Lebih Komprehensif

- Tracking metrik per layer/kelas
- History lengkap untuk semua metrik
- Visualisasi trend metrik dengan highlight model terbaik
- Ekspor ke berbagai format (CSV, JSON, TensorBoard)

### 2. Checkpoint yang Lebih Kaya

- Checkpoint berisi lebih dari sekedar state model
- State optimizer dan scheduler disimpan
- History metrik disimpan untuk analisis dan resume
- Konfigurasi model dan training disimpan untuk reproduktibilitas

### 3. Log yang Lebih Informatif

- Progress log dengan representasi visual (tqdm)
- Log yang diatur dengan emoji kontekstual (✅ ⚠️ ❌ 🚀)
- Highlight nilai metrik (hijau untuk perbaikan, merah untuk regresi)
- Estimasi waktu tersisa yang akurat

## Integrasi dengan Komponen Lain

Implementasi baru terintegrasi dengan baik dengan komponen utils lainnya:

- **Logger**: Pelaporan progress dan metrik yang kaya
- **Visualization**: Visualisasi metrik dan hasil terintegrasi
- **MetricsCalculator**: Perhitungan metrik yang lebih akurat dan lengkap
- **ModelHandler**: Integrasi dengan berbagai arsitektur model

## Panduan Singkat Migrasi

1. **Config**: Transformasi parameter fungsi menjadi dictionary konfigurasi
2. **Inisialisasi**: Buat instance TrainingPipeline dengan konfigurasi
3. **Callbacks**: Tentukan callbacks kustom jika diperlukan
4. **Training**: Jalankan pipeline.train() dengan parameter yang sesuai
5. **Analisis**: Gunakan results dan metrics.history untuk analisis

## Tips Optimasi Training

### 1. Memanfaatkan Callback untuk Ekstensi

```python
# Callback untuk adaptive learning rate
def adaptive_lr_callback(epoch, metrics, **kwargs):
    # Kurangi learning rate jika plateau terdeteksi
    if epoch > 5 and metrics['val_loss'] > prev_loss * 0.99:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.8
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"🔄 Mengurangi LR: {current_lr:.6f} -> {new_lr:.6f}")
    
    # Simpan loss sekarang untuk perbandingan
    global prev_loss
    prev_loss = metrics['val_loss']

# Register callback
pipeline.register_callback('epoch_end', adaptive_lr_callback)
```

### 2. Custom Metrics Calculation

```python
# Definisikan fungsi perhitungan metrik kustom
def compute_f2_score(predictions, targets):
    from sklearn.metrics import fbeta_score
    return fbeta_score(targets, predictions, beta=2, average='weighted')

# Kustomisasi validation epoch dengan metrik tambahan
def custom_metrics_callback(epoch, metrics, **kwargs):
    val_predictions = model(val_images).argmax(dim=1)
    f2 = compute_f2_score(val_predictions.cpu(), val_targets.cpu())
    
    # Tambahkan ke dictionary metrik
    metrics['f2_score'] = f2
    
    # Update history
    pipeline.metrics.update_history('f2_score', f2)
    
    logger.info(f"📊 F2-Score: {f2:.4f}")

# Register callback
pipeline.register_callback('validation_end', custom_metrics_callback)
```

### 3. Memory Management

```python
# Callback untuk memory management
def memory_cleanup_callback(**kwargs):
    # Bersihkan cache CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Log penggunaan memori
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"💾 GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

# Register untuk cleanup setelah setiap epoch
pipeline.register_callback('epoch_end', memory_cleanup_callback)
```

## Contoh Proyek Nyata

Berikut contoh bagaimana training pipeline baru terintegrasi dalam proyek nyata:

```python
"""
Script training: EfficientNet-B4 Backbone untuk Deteksi Mata Uang
"""
from smartcash.utils.logger import get_logger
from smartcash.utils.training import TrainingPipeline
from smartcash.models import ModelHandler, create_efficient_detector
from smartcash.data import DataManager
from smartcash.utils.visualization import MetricsVisualizer

import yaml
import torch
import argparse
import os
from pathlib import Path

# Setup argparse
parser = argparse.ArgumentParser(description="Training SmartCash Currency Detector")
parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config file")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resume")
parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override epochs jika ditentukan
if args.epochs:
    config['training']['epochs'] = args.epochs

# Setup logger
logger = get_logger("training")
logger.start(f"🚀 Memulai training dengan konfigurasi: {args.config}")

# Setup output dir
output_dir = Path(config.get('output_dir', 'runs/train'))
output_dir.mkdir(parents=True, exist_ok=True)

# Inisialisasi data manager
data_manager = DataManager(
    data_dir=config['data']['dir'],
    batch_size=config['training']['batch_size'],
    num_workers=config['model'].get('workers', 4)
)

# Inisialisasi model handler
model_handler = ModelHandler(config)

# Setup metrics visualizer
metrics_vis = MetricsVisualizer(output_dir=output_dir / 'metrics')

# Inisialisasi pipeline
pipeline = TrainingPipeline(
    config=config,
    model_handler=model_handler,
    data_manager=data_manager,
    logger=logger
)

# Callback untuk visualisasi metrik
def visualize_metrics_callback(epoch, metrics, **kwargs):
    if epoch % 5 == 0 or kwargs.get('is_last', False):
        # Plot metrik training
        history = pipeline.metrics.get_all_history()
        fig = metrics_vis.plot_training_metrics(
            history,
            title=f"Training Metrics - Backbone: {config['model']['backbone']}",
            filename=f"metrics_epoch_{epoch}.png"
        )
        
        # Plot confusion matrix jika tersedia
        if hasattr(pipeline, 'last_confusion_matrix'):
            metrics_vis.plot_confusion_matrix(
                pipeline.last_confusion_matrix,
                class_names=data_manager.get_class_names(),
                title=f"Confusion Matrix - Epoch {epoch}",
                filename=f"confusion_matrix_epoch_{epoch}.png"
            )

# Register callbacks
pipeline.register_callback('epoch_end', visualize_metrics_callback)

# Register callback untuk end of training
def final_metrics_callback(epochs_trained, best_val_loss, training_history, **kwargs):
    # Visualisasi akhir
    visualize_metrics_callback(epochs_trained, {}, is_last=True)
    
    # Simpan ringkasan hasil
    summary = (
        f"# Ringkasan Training\n\n"
        f"- Model: {config['model']['backbone']}\n"
        f"- Epochs: {epochs_trained}\n"
        f"- Best Val Loss: {best_val_loss:.4f}\n"
        f"- Dataset: {config['data']['dir']}\n"
        f"- Batch Size: {config['training']['batch_size']}\n"
    )
    
    # Simpan ke file markdown
    with open(output_dir / 'training_summary.md', 'w') as f:
        f.write(summary)
    
    logger.success(f"✅ Hasil training disimpan di {output_dir}")

pipeline.register_callback('training_end', final_metrics_callback)

# Jalankan training
try:
    results = pipeline.train(
        resume_from_checkpoint=args.resume,
        save_every=config['training'].get('save_every', 5)
    )
    
    # Print ringkasan hasil
    logger.success(
        f"✨ Training selesai! Epoch: {results['epochs_trained']}, "
        f"Best val loss: {results['best_val_loss']:.4f}"
    )
    
    if results.get('early_stopped', False):
        logger.info("⏹️ Training berhenti karena early stopping")
        
except KeyboardInterrupt:
    logger.warning("⚠️ Training dihentikan manual dengan keyboard interrupt")
except Exception as e:
    logger.error(f"❌ Error saat training: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Simpan history metrik selalu
    pipeline.metrics.save_to_json(output_dir / 'metrics_history.json')
    logger.info("💾 History metrik disimpan")
```

## Kesimpulan

Implementasi training pipeline baru menawarkan fondasi yang lebih kuat, fleksibel, dan mudah dikelola untuk proses training SmartCash. Dengan pendekatan berorientasi objek dan pemisahan concern, pipeline dapat dengan mudah dikembangkan untuk mendukung berbagai model, dataset, dan skenario training tanpa perlu memodifikasi kode inti.

Manfaat paling signifikan adalah sistem callback yang memberikan fleksibilitas luar biasa untuk memperluas fungsionalitas pipeline serta pengelolaan metrik yang lebih baik untuk analisis performa model yang lebih mendalam. Kemampuan resume training dan early stopping yang ditingkatkan juga mengurangi waktu dan resource yang diperlukan untuk eksperimen.

Dengan pipeline training baru ini, SmartCash dapat lebih efisien dalam pengembangan model dan eksperimen berbasis data, memungkinkan iterasi yang lebih cepat dan hasil yang lebih dapat direproduksi.