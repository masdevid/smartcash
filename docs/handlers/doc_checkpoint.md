# Dokumentasi CheckpointManager

## Gambaran Umum

`CheckpointManager` adalah komponen utama untuk mengelola proses checkpoint pada model SmartCash. Komponen ini dirancang dengan pendekatan modular untuk menangani berbagai aspek pengelolaan checkpoint seperti penyimpanan, pemuatan, pencarian, dan pengelolaan riwayat training.

## Struktur Komponen

Sistem checkpoint terdiri dari beberapa komponen yang saling bekerja sama:

```
smartcash/handlers/checkpoint/
├── __init__.py                  # Export komponen utama
├── checkpoint_manager.py        # Facade utama (entry point)
├── checkpoint_loader.py         # Pemuatan dan validasi checkpoint
├── checkpoint_saver.py          # Penyimpanan dan backup checkpoint
├── checkpoint_finder.py         # Pencarian dan filter checkpoint
├── checkpoint_history.py        # Pengelolaan riwayat training
└── checkpoint_utils.py          # Fungsi utilitas umum
```

## Penggunaan Dasar

### Inisialisasi Manager

```python
from smartcash.handlers.checkpoint import CheckpointManager
from smartcash.utils.logger import get_logger

# Inisialisasi dengan logger kustom (opsional)
logger = get_logger("training")

# Buat instance checkpoint manager
checkpoint_manager = CheckpointManager(
    output_dir="runs/train/weights",
    logger=logger
)
```

### Menyimpan Checkpoint

```python
# Menyimpan checkpoint di akhir epoch
checkpoint_result = checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    epoch=current_epoch,
    metrics={'loss': val_loss, 'accuracy': val_accuracy},
    is_best=(val_loss < best_val_loss)
)

# Dapatkan path checkpoint yang disimpan
checkpoint_path = checkpoint_result['path']
print(f"✅ Checkpoint disimpan ke: {checkpoint_path}")
```

### Memuat Checkpoint

```python
# Memuat checkpoint terbaik
checkpoint = checkpoint_manager.load_checkpoint(
    model=model,              # Optional: Langsung muat state ke model
    optimizer=optimizer,      # Optional: Langsung muat state ke optimizer
    scheduler=scheduler       # Optional: Langsung muat state ke scheduler
)

# Dapatkan epoch terakhir untuk resume training
last_epoch = checkpoint.get('epoch', 0)
print(f"📂 Melanjutkan training dari epoch {last_epoch}")
```

### Mencari Checkpoint

```python
# Mencari checkpoint terbaik
best_checkpoint_path = checkpoint_manager.find_best_checkpoint()

# Mencari checkpoint terakhir
latest_checkpoint_path = checkpoint_manager.find_latest_checkpoint()

# Mencari checkpoint berdasarkan epoch
epoch10_checkpoint = checkpoint_manager.find_checkpoint_by_epoch(10)

# Filter checkpoint berdasarkan karakteristik
filtered_checkpoints = checkpoint_manager.filter_checkpoints(
    backbone="efficientnet_b4",  # Filter berdasarkan backbone
    dataset="roboflow",         # Filter berdasarkan dataset
    min_epoch=5                 # Filter min epoch
)
```

## Fitur Lanjutan

### Membersihkan Checkpoint yang Tidak Digunakan

```python
# Membersihkan checkpoint lama
deleted_paths = checkpoint_manager.cleanup_checkpoints(
    max_checkpoints=5,    # Jumlah maks checkpoint per kategori
    keep_best=True,       # Pertahankan semua checkpoint terbaik
    keep_latest=True,     # Pertahankan checkpoint latest terakhir
    max_epochs=5          # Jumlah maks checkpoint epoch
)

print(f"🧹 {len(deleted_paths)} checkpoint lama dibersihkan")
```

### Menyalin ke Google Drive (untuk Colab)

```python
# Salin checkpoint ke Google Drive
copied_paths = checkpoint_manager.copy_to_drive(
    drive_dir="/content/drive/MyDrive/SmartCash/checkpoints",
    best_only=True  # Hanya salin checkpoint terbaik
)

print(f"📤 {len(copied_paths)} checkpoint disalin ke Drive")
```

### Riwayat Training & Ekspor

```python
# Dapatkan riwayat training
history = checkpoint_manager.get_training_history()
print(f"📊 Total sesi training: {history['total_runs']}")

# Ekspor riwayat ke JSON
json_path = checkpoint_manager.export_history_to_json("training_history.json")
print(f"💾 Riwayat training disimpan ke {json_path}")
```

### Menampilkan Daftar Checkpoint

```python
# Tampilkan daftar checkpoint yang tersedia
checkpoint_manager.display_checkpoints()
```

### Validasi Checkpoint

```python
# Validasi apakah checkpoint valid
is_valid, validation_info = checkpoint_manager.validate_checkpoint("path/to/checkpoint.pth")
if is_valid:
    print("✅ Checkpoint valid dan lengkap")
else:
    print(f"❌ Checkpoint tidak valid: {validation_info['message']}")
```

## Rekomendasi Penggunaan

1. **Penyimpanan Checkpoint Reguler**:
   - Simpan checkpoint di setiap akhir epoch
   - Manfaatkan flag `is_best` untuk menandai checkpoint terbaik

2. **Mengelola Ukuran Disk**:
   - Gunakan `cleanup_checkpoints()` secara berkala untuk membersihkan checkpoint lama
   - Sebaiknya panggil setiap ~10 epoch

3. **Resume Training**:
   - Selalu gunakan `load_checkpoint()` untuk melanjutkan training yang terhenti
   - Manfaatkan metadata yang tersimpan di checkpoint

4. **Backup Checkpoint**:
   - Gunakan `copy_to_drive()` secara berkala untuk mencadangkan checkpoint penting

## Troubleshooting

### Checkpoint Tidak Ditemukan

- Periksa direktori output: `checkpoint_manager.output_dir`
- Pastikan direktori tersebut memiliki izin yang tepat
- Coba gunakan `list_checkpoints()` untuk melihat semua checkpoint yang tersedia

### Gagal Memuat Model State

- Coba gunakan `validate_checkpoint()` untuk memeriksa integritas file checkpoint
- Jika struktur model berubah, gunakan parameter `strict=False` pada `model.load_state_dict()`

### Riwayat Training Tidak Lengkap

- Periksa file `training_history.yaml` di direktori checkpoint
- Gunakan `export_history_to_json()` untuk mengekspor riwayat ke format yang lebih mudah dibaca

## Catatan Penting

- Semua operasi pada CheckpointManager sudah thread-safe
- Semua fungsi penyimpanan dilengkapi dengan penanganan error yang robust
- Gunakan parameter `logger` untuk mendapatkan log yang lebih detail

Dengan menggunakan CheckpointManager dengan benar, Anda dapat mengelola proses checkpoint secara efisien dan mengurangi risiko kehilangan hasil training yang berharga.