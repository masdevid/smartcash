# Dokumentasi CheckpointManager SmartCash

## Deskripsi

`CheckpointManager` adalah komponen pusat untuk pengelolaan checkpoint model deteksi mata uang Rupiah di SmartCash. 
Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi checkpoint. Implementasi telah dioptimasi dengan pendekatan yang modular dan mudah dipelihara.

## Struktur dan Komponen

`CheckpointManager` mengadopsi struktur modular berikut:

```
smartcash/handlers/checkpoint/
├── __init__.py                     # Export komponen utama
├── checkpoint_manager.py           # Entry point minimal (facade)
├── checkpoint_loader.py            # Loading checkpoint model
├── checkpoint_saver.py             # Penyimpanan checkpoint model
├── checkpoint_finder.py            # Pencarian checkpoint
├── checkpoint_history.py           # Pengelolaan riwayat training
└── checkpoint_utils.py             # Utilitas umum
```

`CheckpointManager` menggabungkan beberapa komponen terspesialisasi menjadi satu antarmuka terpadu:

- **CheckpointLoader**: Loading model dan state dari checkpoint
- **CheckpointSaver**: Penyimpanan model ke checkpoint
- **CheckpointFinder**: Pencarian checkpoint berdasarkan kriteria
- **CheckpointHistory**: Pengelolaan riwayat checkpoint dan training

## Fitur Utama

### 1. Pengelolaan Checkpoint Model

- Loading dan penyimpanan model dengan metadata komprehensif
- Support untuk menyimpan state optimizer dan scheduler
- Loading model dengan atau tanpa strict mode
- Validasi checkpoint dan validasi kompatibilitas model

### 2. Organisasi Checkpoint Training

- Penyimpanan berbagai jenis checkpoint (best, latest, epoch)
- Pengelolaan riwayat training dengan metadata lengkap
- Penyusunan struktur checkpoint yang konsisten dan robust
- Recovery otomatis dari error dengan emergency checkpoint

### 3. Pencarian Checkpoint

- Pencarian checkpoint terbaik berdasarkan metrik
- Pencarian checkpoint untuk epoch tertentu
- Pencarian checkpoint terbaru berdasarkan waktu modifikasi
- Pencarian berdasarkan backbone dan konfigurasi lainnya

### 4. Pengelolaan History Training

- Pencatatan riwayat training secara komprehensif
- Dukungan untuk resume training dari checkpoint tertentu
- Pelacakan checkpoint terbaik berdasarkan metrik

### 5. Optimasi Penyimpanan

- Pembersihan checkpoint lama secara otomatis
- Pembatasan jumlah checkpoint berdasarkan tipe
- Menyimpan checkpoint secara periodik berdasarkan epoch

### 6. Integrasi dengan Google Colab

- Dukungan untuk menyalin checkpoint ke Google Drive
- Deteksi otomatis environment Google Colab
- Penyimpanan riwayat training dan loading dari Drive

## Kelas Utama

### CheckpointManager

Manager utama untuk pengelolaan checkpoint model SmartCash. Mengimplementasikan pola Facade untuk menyediakan API terpadu ke semua komponen terkait checkpoint.

**Parameter Init:**
- `output_dir`: Direktori untuk menyimpan checkpoint
- `logger`: Logger kustom (opsional)

### CheckpointLoader

Komponen untuk loading model checkpoint dengan penanganan error yang robust.

**Metode Utama:**
- `load_checkpoint()`: Muat checkpoint dengan dukungan berbagai opsi dan resume training

### CheckpointSaver

Komponen untuk penyimpanan model checkpoint dengan penanganan error yang robust.

**Metode Utama:**
- `save_checkpoint()`: Simpan checkpoint model dengan metadata komprehensif

### CheckpointFinder

Pencarian checkpoint model berdasarkan berbagai kriteria.

**Metode Utama:**
- `find_best_checkpoint()`: Temukan checkpoint terbaik berdasarkan riwayat training
- `find_latest_checkpoint()`: Temukan checkpoint terakhir berdasarkan waktu modifikasi

### CheckpointHistory

Pengelolaan riwayat checkpoint dan training.

**Metode Utama:**
- `update_training_history()`: Update riwayat training dalam file YAML

## Format Metadata Checkpoint

Checkpoint SmartCash menyimpan metadata komprehensif untuk memudahkan resume training dan evaluasi:

```python
{
    'epoch': 30,                      # Epoch terakhir
    'model_state_dict': {...},        # State dict model
    'optimizer_state_dict': {...},    # State dict optimizer (opsional)
    'scheduler_state_dict': {...},    # State dict scheduler (opsional)
    'metrics': {                      # Metrik training
        'loss': 0.234,
        'val_loss': 0.345,
        'mAP': 0.678,
        'precision': 0.765,
    },
    'config': {                       # Konfigurasi training
        'model': {...},
        'training': {...},
    },
    'timestamp': '2023-05-20T15:30:45.123456',  # Waktu penyimpanan
}
```

## Konvensi Penamaan Checkpoint

Checkpoint menggunakan konvensi penamaan yang konsisten:

```
smartcash_{backbone}_{dataset}_{type}_{timestamp}.pth
```

Contoh:
- `smartcash_efficientnet_b4_roboflow_best_20230520_153045.pth`
- `smartcash_cspdarknet_local_latest_20230520_153045.pth`
- `smartcash_efficientnet_b4_roboflow_epoch_15_20230520_153045.pth`

## Format History Training

History training disimpan dalam format YAML:

```yaml
total_runs: 45
runs:
  - checkpoint_name: "smartcash_efficientnet_b4_roboflow_best_20230520_153045.pth"
    timestamp: "2023-05-20T15:30:45.123456"
    is_best: true
    epoch: 30
    metrics:
      loss: 0.234
      val_loss: 0.345
      mAP: 0.678
      precision: 0.765
last_resume:
  checkpoint: "smartcash_efficientnet_b4_roboflow_best_20230520_153045.pth"
  timestamp: "2023-05-21T09:15:30.654321"
```

## Metode Utama di CheckpointManager

### load_checkpoint

Memuat checkpoint model beserta state optimizer dan scheduler jika disediakan. Jika checkpoint_path tidak diberikan, akan mencari checkpoint terbaik berdasarkan riwayat training.

**Parameter:**
- `checkpoint_path`: Path ke checkpoint (jika None, akan mengambil checkpoint terbaik)
- `device`: Perangkat untuk memuat model
- `model`: Model yang akan dimuat dengan weights (opsional)
- `optimizer`: Optimizer yang akan dimuat dengan state (opsional)
- `scheduler`: Scheduler yang akan dimuat dengan state (opsional)

### save_checkpoint

Menyimpan checkpoint model beserta metadata lengkap. Flag `is_best` menentukan apakah checkpoint akan disimpan sebagai checkpoint terbaik.

**Parameter:**
- `model`: Model PyTorch
- `optimizer`: Optimizer
- `scheduler`: Learning rate scheduler
- `config`: Konfigurasi training
- `epoch`: Epoch saat ini
- `metrics`: Metrik training
- `is_best`: Apakah ini model terbaik
- `save_optimizer`: Apakah menyimpan state optimizer

### find_best_checkpoint

Mencari checkpoint terbaik berdasarkan riwayat training. Checkpoint terbaik ditentukan oleh flag `is_best` yang disimpan dalam history.

### find_latest_checkpoint

Mencari checkpoint terakhir berdasarkan waktu modifikasi file.

### find_checkpoint_by_epoch

Mencari checkpoint untuk epoch tertentu.

**Parameter:**
- `epoch`: Nomor epoch yang dicari

### list_checkpoints

Mendapatkan daftar semua checkpoint yang tersedia, dikelompokkan berdasarkan tipe (best, latest, epoch).

### cleanup_checkpoints

Membersihkan checkpoint lama berdasarkan kriteria tertentu untuk menghemat ruang penyimpanan.

**Parameter:**
- `max_checkpoints`: Jumlah maksimal checkpoint per kategori
- `keep_best`: Pertahankan semua checkpoint terbaik
- `keep_latest`: Pertahankan checkpoint latest terakhir
- `max_epochs`: Jumlah maksimal checkpoint epoch yang disimpan

### get_training_history

Mendapatkan riwayat training lengkap dari file history.

## Integrasi dengan Komponen Lain

### 1. Integrasi dengan ModelManager

CheckpointManager dapat digunakan dengan ModelManager untuk:
- Loading model dari checkpoint terbaik
- Menyimpan checkpoint selama training
- Mengelola riwayat training dan validasi

### 2. Integrasi dengan EvaluationManager

CheckpointManager dapat diintegrasikan dengan EvaluationManager untuk:
- Evaluasi model dari checkpoint terbaik
- Perbandingan performa berbagai checkpoint
- Analisis kemajuan training berdasarkan metrik

### 3. Integrasi dengan Google Colab

Dukungan khusus untuk environment Google Colab:
- Deteksi otomatis environment Colab
- Menyalin checkpoint terbaik ke Google Drive
- Mengakses checkpoint dari Google Drive

## Penanganan Error

CheckpointManager memiliki mekanisme robust untuk menangani berbagai error:

1. **Recovery dari Invalid Checkpoint**:
   - Jika checkpoint terbaik tidak valid, gunakan checkpoint terakhir
   - Jika tidak ada checkpoint yang valid, mulai dari awal
   
2. **Non-strict Loading**:
   - Mendukung loading model dengan `strict=False` jika terjadi error
   - Memungkinkan loading partial state untuk transfer learning

3. **Emergency Checkpoint**:
   - Menyimpan emergency checkpoint jika operasi save normal gagal
   - Mempertahankan minimal state model untuk menyelamatkan progress

4. **Fallback Mechanism**:
   - Fallback ke default values jika metadata tidak lengkap
   - Memungkinkan recovery dari checkpoint yang rusak atau tidak lengkap

## Kesimpulan

CheckpointManager SmartCash menawarkan:

1. **Antarmuka Terpadu**: Menyederhanakan operasi checkpoint yang kompleks
2. **Keandalan Tinggi**: Penanganan error robust dan mekanisme recovery
3. **Modularitas**: Pemisahan tanggung jawab ke komponen terpisah
4. **Pengelolaan Riwayat**: Pelacakan training history yang komprehensif
5. **Integrasi Mulus**: Bekerja seamless dengan komponen lain di SmartCash
6. **Colab Support**: Dukungan khusus untuk Google Colab dan Drive

Dengan fitur-fitur tersebut, CheckpointManager memastikan proses training dapat dilanjutkan dengan mulus, progress training terrekam dengan baik, dan model terbaik dapat dengan mudah ditemukan untuk evaluasi dan deployment.