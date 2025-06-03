"""
File: smartcash/ui/pretrained_model/constants/model_constants.py
Deskripsi: Single source constants untuk model URLs dan configurations
"""

# Model configurations sebagai single source of truth
MODEL_CONFIGS = {
    'yolov5': {
        'name': 'YOLOv5s',
        'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
        'filename': 'yolov5s.pt',
        'min_size_mb': 10,
        'description': 'Object detection backbone'
    },
    'efficientnet-b4': {
        'name': 'EfficientNet-B4',
        'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
        'filename': 'efficientnet_b4_huggingface.bin',
        'min_size_mb': 60,
        'description': 'Feature extraction backbone'
    }
}

# Default paths
DEFAULT_MODELS_DIR = '/content/models'
DEFAULT_DRIVE_MODELS_DIR = '/content/drive/MyDrive/SmartCash/models'

# Progress step definitions
PROGRESS_STEPS = {
    'INIT': (0, 'Inisialisasi'),
    'AUTO_CHECK': (10, 'Auto-check model existing'),
    'CHECK_MODELS': (20, 'Memeriksa model tersedia'),
    'DOWNLOAD_START': (30, 'Memulai download'),
    'DOWNLOAD_PROGRESS': (50, 'Mengunduh model'),
    'SYNC_START': (80, 'Memulai sinkronisasi'),
    'SYNC_COMPLETE': (95, 'Sinkronisasi selesai'),
    'COMPLETE': (100, 'Proses selesai')
}