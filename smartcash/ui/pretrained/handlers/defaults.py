"""
File: smartcash/ui/pretrained/handlers/defaults.py
Deskripsi: Default configuration untuk pretrained module
"""

from typing import Dict, Any

def get_default_pretrained_config() -> Dict[str, Any]:
    """
    Get default configuration untuk pretrained module.
    
    Returns:
        Dict berisi default pretrained configuration
    """
    return {
        'pretrained_models': {
            # Direktori models (updated ke /data/pretrained)
            'models_dir': '/data/pretrained',
            'drive_models_dir': '/content/drive/MyDrive/SmartCash/pretrained',
            
            # Model configurations
            'models': {
                'yolov5s': {
                    'name': 'YOLOv5s',
                    'source': 'ultralytics',
                    'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                    'filename': 'yolov5s.pt',
                    'min_size_mb': 10,
                    'description': 'YOLOv5s object detection backbone dari Ultralytics',
                    'architecture': 'CSPDarknet',
                    'input_size': [640, 640],
                    'output_channels': [128, 256, 512]
                },
                'efficientnet_b4': {
                    'name': 'EfficientNet-B4',
                    'source': 'timm',
                    'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                    'filename': 'efficientnet_b4_timm.bin',
                    'min_size_mb': 60,
                    'description': 'EfficientNet-B4 feature extraction backbone dari TIMM',
                    'architecture': 'EfficientNet',
                    'input_size': [640, 640],
                    'output_channels': [56, 160, 448]  # Akan di-adapt ke [128, 256, 512]
                }
            }
        },
        
        # Progress tracking configuration
        'progress': {
            'steps': {
                'init': {'value': 0, 'label': 'Inisialisasi'},
                'check_models': {'value': 20, 'label': 'Memeriksa model tersedia'},
                'download_start': {'value': 30, 'label': 'Memulai download'},
                'download_progress': {'value': 50, 'label': 'Mengunduh model'},
                'sync_start': {'value': 80, 'label': 'Memulai sinkronisasi'},
                'sync_complete': {'value': 95, 'label': 'Sinkronisasi selesai'},
                'complete': {'value': 100, 'label': 'Proses selesai'}
            }
        },
        
        # Cache configuration
        'cache': {
            'dir': '.cache/smartcash/pretrained',
            'auto_cleanup': True,
            'max_size_gb': 2
        },
        
        # Download configuration
        'download': {
            'timeout': 300,  # 5 minutes
            'chunk_size': 8192,
            'retry_attempts': 3,
            'verify_ssl': True
        }
    }

def get_model_variants() -> Dict[str, Dict[str, Any]]:
    """
    Get available model variants untuk customization.
    
    Returns:
        Dict berisi model variants yang bisa dipilih user
    """
    return {
        'yolov5_variants': {
            'yolov5n': {
                'name': 'YOLOv5n (Nano)',
                'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt',
                'filename': 'yolov5n.pt',
                'min_size_mb': 2,
                'description': 'Smallest and fastest YOLOv5 model'
            },
            'yolov5s': {
                'name': 'YOLOv5s (Small)',
                'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt',
                'filename': 'yolov5s.pt',
                'min_size_mb': 10,
                'description': 'Small model with good balance of speed and accuracy'
            },
            'yolov5m': {
                'name': 'YOLOv5m (Medium)',
                'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt',
                'filename': 'yolov5m.pt',
                'min_size_mb': 40,
                'description': 'Medium model with better accuracy'
            },
            'yolov5l': {
                'name': 'YOLOv5l (Large)',
                'url': 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt',
                'filename': 'yolov5l.pt',
                'min_size_mb': 90,
                'description': 'Large model with high accuracy'
            }
        },
        
        'efficientnet_variants': {
            'efficientnet_b0': {
                'name': 'EfficientNet-B0',
                'url': 'https://huggingface.co/timm/efficientnet_b0.ra_in1k/resolve/main/pytorch_model.bin',
                'filename': 'efficientnet_b0_timm.bin',
                'min_size_mb': 20,
                'description': 'Smallest EfficientNet model'
            },
            'efficientnet_b4': {
                'name': 'EfficientNet-B4',
                'url': 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin',
                'filename': 'efficientnet_b4_timm.bin',
                'min_size_mb': 60,
                'description': 'Balanced performance and accuracy'
            },
            'efficientnet_b7': {
                'name': 'EfficientNet-B7',
                'url': 'https://huggingface.co/timm/efficientnet_b7.ra2_in1k/resolve/main/pytorch_model.bin',
                'filename': 'efficientnet_b7_timm.bin',
                'min_size_mb': 250,
                'description': 'Largest EfficientNet model with highest accuracy'
            }
        }
    }