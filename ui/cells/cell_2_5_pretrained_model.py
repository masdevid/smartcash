"""
File: smartcash/ui/cells/cell_2_5_pretrained_model.py
Deskripsi: Download model pre-trained YOLOv5 dan EfficientNet-B4 untuk SmartCash dengan sinkronisasi Google Drive
"""

# Import utilitas sinkronisasi Drive
from smartcash.common.drive_sync import sync_models_with_drive

# Sinkronisasi dari Drive terlebih dahulu
MODELS_DIR = '/content/models'
DRIVE_MODELS_DIR = '/content/drive/MyDrive/SmartCash/models'

# Lakukan sinkronisasi awal dari Drive
sync_models_with_drive(MODELS_DIR, DRIVE_MODELS_DIR)

# Import setelah sinkronisasi
from smartcash.model.services.pretrained_setup import setup_pretrained_models

# Eksekusi download model
model_info = setup_pretrained_models(models_dir=MODELS_DIR)

# Sinkronisasi ke Drive setelah download
if model_info:
    sync_models_with_drive(MODELS_DIR, DRIVE_MODELS_DIR, model_info)
