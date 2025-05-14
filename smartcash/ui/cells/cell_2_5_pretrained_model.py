"""
File: smartcash/ui/cells/cell_2_5_pretrained_model.py
Deskripsi: Download model pre-trained YOLOv5 dan EfficientNet-B4 untuk SmartCash dengan sinkronisasi Google Drive
"""

import os
from pathlib import Path

# Import utilitas sinkronisasi Drive
from smartcash.common.drive_sync import sync_models_with_drive

# Definisikan direktori untuk model
MODELS_DIR = '/content/models'
DRIVE_MODELS_DIR = '/content/drive/MyDrive/SmartCash/models'

# Cek apakah direktori parent ada
models_parent = Path(MODELS_DIR).parent
if not models_parent.exists():
    print(f"⚠️ Direktori parent {models_parent} tidak ditemukan")
    print("⚠️ Download dan sinkronisasi model dilewati")
    exit()

# Cek apakah berjalan di Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Cek apakah Drive tersedia jika di Colab
if IN_COLAB and not os.path.exists('/content/drive'):
    try:
        from google.colab import drive
        print("🔄 Mounting Google Drive...")
        drive.mount('/content/drive')
    except Exception as e:
        print(f"❌ Gagal mounting Google Drive: {str(e)}")
        print("⚠️ Sinkronisasi dengan Drive dilewati, tetapi download tetap dilanjutkan")

# Buat direktori model jika belum ada
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# Lakukan sinkronisasi awal dari Drive jika di Colab
if IN_COLAB and os.path.exists('/content/drive'):
    sync_models_with_drive(MODELS_DIR, DRIVE_MODELS_DIR)

# Import setelah sinkronisasi
from smartcash.model.services.pretrained_setup import setup_pretrained_models

# Eksekusi download model
model_info = setup_pretrained_models(models_dir=MODELS_DIR)

# Sinkronisasi ke Drive setelah download jika di Colab
if model_info and IN_COLAB and os.path.exists('/content/drive'):
    sync_models_with_drive(MODELS_DIR, DRIVE_MODELS_DIR, model_info)
