"""
File: smartcash/ui/dataset/utils/format_mapping.py
Deskripsi: Utilitas untuk memetakan format dataset UI ke format yang didukung oleh library Roboflow
"""

from typing import Dict, Optional

# Mapping format UI ke format Roboflow
FORMAT_MAPPING = {
    'YOLO v5': 'yolov5pytorch',  # Standardisasi ke format yolov5pytorch
    'COCO': 'coco',
    'VOC': 'voc',
    'YOLO v8': 'yolov8',
    'TensorFlow': 'tensorflow',
    'TensorFlow Object Detection': 'tfrecord'
}

def get_roboflow_format(ui_format: str) -> str:
    """
    Memetakan format UI ke format Roboflow yang sesuai.
    
    Args:
        ui_format: Format yang dipilih dari UI
        
    Returns:
        Format Roboflow yang sesuai
    """
    # Jika format tidak ada di mapping, gunakan yolov5pytorch sebagai default
    return FORMAT_MAPPING.get(ui_format, 'yolov5pytorch')

def get_ui_formats() -> Dict[str, str]:
    """
    Dapatkan seluruh mapping format untuk UI dropdown.
    
    Returns:
        Dictionary berisi mapping format UI ke format Roboflow
    """
    return FORMAT_MAPPING

def get_format_description(format_name: str) -> str:
    """
    Dapatkan deskripsi format berdasarkan nama format.
    
    Args:
        format_name: Nama format UI
        
    Returns:
        Deskripsi format
    """
    descriptions = {
        'YOLO v5': 'Format PyTorch untuk YOLO v5 (*.txt dan folder images/labels)',
        'YOLO v8': 'Format untuk YOLO v8 dari Ultralytics',
        'COCO': 'Format JSON standar COCO dataset',
        'VOC': 'Format XML Pascal VOC',
        'TensorFlow': 'Format TFRecord untuk TensorFlow',
        'TensorFlow Object Detection': 'Format TFRecord untuk TensorFlow Object Detection API'
    }
    
    return descriptions.get(format_name, 'Format dataset')