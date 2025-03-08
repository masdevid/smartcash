# Dokumentasi Model Handler SmartCash

## Ringkasan
Model Handler SmartCash telah direfaktorisasi menjadi komponen yang lebih modular dan terorganisir berdasarkan prinsip *Single Responsibility*. Refaktorisasi ini meningkatkan pemeliharaan kode, mendukung pengujian yang lebih baik, dan memanfaatkan komponen `utils/training` yang telah direfaktorisasi sebelumnya.

## Struktur Komponen
```
smartcash/
├── handlers/
│   ├── model_manager.py          # Entry point utama (facade)
│   └── model/
│       ├── __init__.py           # Export semua komponen
│       ├── model_factory.py      # Pembuatan model dengan berbagai backbone
│       ├── optimizer_factory.py  # Pembuatan optimizer dan scheduler
│       ├── model_trainer.py      # Training model
│       ├── model_evaluator.py    # Evaluasi model
│       ├── model_predictor.py    # Prediksi dengan model terlatih
│       └── model_experiments.py  # Eksperimen dan hyperparameter tuning
```

## Entry Point: ModelManager
`ModelManager` adalah facade yang menyediakan akses terpadu ke semua komponen model. Kelas ini terletak di `handlers/model_manager.py` dan menyediakan interface yang bersih untuk semua operasi model.

## Penggunaan

### Inisialisasi
```python
from smartcash.handlers.model_manager import ModelManager

# Inisialisasi dengan konfigurasi
model_manager = ModelManager(config={
    'model': {
        'backbone': 'efficientnet',
        'num_classes': 7
    },
    'training': {
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001
    },
    'output_dir': 'runs/training'
})
```

### Training Model
```python
# Dapatkan dataloaders
train_loader = data_manager.get_train_loader()
val_loader = data_manager.get_val_loader()

# Training model
results = model_manager.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    device='cuda'
)

# Akses hasil
best_checkpoint_path = results['best_checkpoint_path']
print(f"Training selesai dengan val_loss: {results['best_val_loss']}")
```

### Evaluasi Model
```python
# Dapatkan test dataloader
test_loader = data_manager.get_test_loader()

# Evaluasi model
metrics = model_manager.evaluate(
    test_loader=test_loader,
    checkpoint_path='path/to/checkpoint.pth'
)

# Print hasil
print(f"Akurasi: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

### Prediksi dengan Model
```python
# Prediksi single image
image = torch.tensor(...)  # [C, H, W]
result = model_manager.predict(
    images=image,
    conf_threshold=0.5
)

# Akses hasil
detections = result['detections']
```

### Pembekuan/Unfreeze Backbone
```python
# Buat model
model = model_manager.create_model(backbone_type='efficientnet')

# Bekukan backbone untuk fine-tuning
model = model_manager.model_factory.freeze_backbone(model)

# Training hanya layer-layer atas
results = model_manager.train(train_loader, val_loader, model=model)

# Lepas pembekuan jika diperlukan
model = model_manager.model_factory.unfreeze_backbone(model)

# Lanjutkan training dengan seluruh model
results = model_manager.train(train_loader, val_loader, model=model)
```

### Eksperimen dengan Backbone Berbeda
```python
# Bandingkan backbone
backbones = ['efficientnet', 'cspdarknet']
results = model_manager.compare_backbones(
    backbones=backbones,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

### Hyperparameter Tuning
```python
# Grid parameter
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [8, 16, 32]
}

# Jalankan tuning
tuning_results = model_manager.tune_hyperparameters(
    param_grid=param_grid,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    max_experiments=5  # Opsional: batasi jumlah eksperimen
)
```

## Komponen Individual

### ModelFactory
Bertanggung jawab untuk membuat model dengan berbagai backbone dan konfigurasi.

```python
from smartcash.handlers.model.model_factory import ModelFactory

factory = ModelFactory(config)
model = factory.create_model(backbone_type='efficientnet')

# Freeze/unfreeze backbone
model = factory.freeze_backbone(model)  # Bekukan layer backbone
model = factory.unfreeze_backbone(model)  # Lepaskan pembekuan
```

### OptimizerFactory
Membuat optimizer dan scheduler berdasarkan konfigurasi.

```python
from smartcash.handlers.model.optimizer_factory import OptimizerFactory

factory = OptimizerFactory(config)
optimizer = factory.create_optimizer(model, lr=0.001)
scheduler = factory.create_scheduler(optimizer)
```

### ModelTrainer
Khusus untuk proses training model menggunakan TrainingPipeline yang direfaktor.

```python
from smartcash.handlers.model.model_trainer import ModelTrainer

trainer = ModelTrainer(config)
results = trainer.train(train_loader, val_loader)

# Resume training dari checkpoint
results = trainer.resume_training(
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_path='path/to/last_checkpoint.pth'
)
```

### ModelEvaluator
Khusus untuk evaluasi model dengan metrik yang komprehensif.

```python
from smartcash.handlers.model.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(config)
metrics = evaluator.evaluate(test_loader, model=model)

# Multiple runs untuk metrik yang lebih stabil
avg_metrics = evaluator.evaluate_multiple_runs(
    test_loader=test_loader,
    checkpoint_path='path/to/best.pth',
    num_runs=3
)
```

### ModelPredictor
Khusus untuk melakukan inferensi/prediksi dengan model terlatih.

```python
from smartcash.handlers.model.model_predictor import ModelPredictor

predictor = ModelPredictor(config)
results = predictor.predict(images, conf_threshold=0.5)

# Prediksi batch
batch_results = predictor.predict_batch(
    dataloader=test_loader,
    conf_threshold=0.5,
    max_samples=100
)
```

### ModelExperiments
Menjalankan eksperimen dan membandingkan hasil dari berbagai konfigurasi.

```python
from smartcash.handlers.model.model_experiments import ModelExperiments

experiments = ModelExperiments(config)

# Single experiment
scenario = {
    'name': 'EfficientNet-Test', 
    'description': 'Test EfficientNet backbone', 
    'backbone': 'efficientnet'
}
results = experiments.run_experiment(scenario, train_loader, val_loader, test_loader)

# Compare different backbones
compare_results = experiments.compare_backbones(
    backbones=['efficientnet', 'cspdarknet'],
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

## Integrasi dengan utils/training
Refaktorisasi ini memanfaatkan komponen dari `utils/training` yang telah direfaktorisasi:

- **TrainingPipeline** - Digunakan dalam `ModelTrainer` untuk mengelola alur training
- **MetricsCalculator** - Digunakan dalam `ModelEvaluator` untuk perhitungan metrik yang akurat
- **Training Callbacks** - Digunakan untuk merespon events training

## Upgrade dari ModelHandler Lama

ModelHandler lama (`handlers/model_handler.py`) tidak lagi digunakan. Sebagai gantinya, gunakan `ModelManager` yang baru sebagai entry point untuk semua fungsionalitas model.

### Contoh Migrasi:

**Kode Lama:**
```python
from smartcash.handlers.model_handler import ModelHandler

model_handler = ModelHandler(config)
model = model_handler.create_model()
results = model_handler.train(train_loader, val_loader)
metrics = model_handler.evaluate(test_loader)
```

**Kode Baru:**
```python
from smartcash.handlers.model_manager import ModelManager

model_manager = ModelManager(config)
model = model_manager.create_model()
results = model_manager.train(train_loader, val_loader)
metrics = model_manager.evaluate(test_loader)
```